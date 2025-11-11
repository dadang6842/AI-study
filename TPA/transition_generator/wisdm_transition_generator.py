import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

class WISDMTransitionDatasetCreator:
    def __init__(self, wisdm_data_path):
        self.wisdm_data_path = Path(wisdm_data_path)
        
        self.label_encoder = LabelEncoder()
        
        self.transitions = [
            ('Standing', 'Sitting'),
            ('Sitting', 'Standing'),
            
            ('Standing', 'Walking'),
            ('Walking', 'Standing'),
            
            ('Walking', 'Jogging'),
            ('Jogging', 'Walking'),
            
            ('Walking', 'Upstairs'),
            ('Walking', 'Downstairs'),
            ('Upstairs', 'Walking'),
            ('Downstairs', 'Walking'),
        ]
        
        self.mixing_ratios = [0.1, 0.2, 0.3, 0.4]
        
        self.timestep = 200
        self.step = 40
        
    def load_wisdm_raw(self):
        column_names = ['user', 'activity', 'timestamp', 'x', 'y', 'z', 'NaN']
        df = pd.read_csv(self.wisdm_data_path, header=None, names=column_names, comment=';')

        df = df.drop('NaN', axis=1).dropna()
        
        df['z'] = df['z'].replace(regex=True, to_replace=r';', value=r'')

        df = df.dropna()
        
        df['activity_code'] = self.label_encoder.fit_transform(df['activity'])
        
        return df
    
    def create_dataset(self, X, y, time_steps=200, step=40):
        xs, ys = [], []
        for i in range(0, len(X) - time_steps, step):
            v = X.iloc[i:i + time_steps].values
            labels = y.iloc[i:i + time_steps]
            values, counts = np.unique(labels, return_counts=True)
            mode_label = values[np.argmax(counts)]
            xs.append(v)
            ys.append(mode_label)
        return np.array(xs), np.array(ys)
    
    def split_data(self, X, y, test_size=0.2):
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_transition_sample(self, from_sample, to_sample, mixing_ratio):
        T, C = from_sample.shape
        
        split_point = int(T * (1 - mixing_ratio))
        
        transition_sample = np.zeros_like(from_sample)
        transition_sample[:split_point, :] = from_sample[:split_point, :]
        transition_sample[split_point:, :] = to_sample[:T - split_point, :]
        
        return transition_sample
    
    def create_dataset_with_transition(self, X, y, from_activity, to_activity,
                                       mixing_ratio, augmentation_ratio):
        from_code = self.label_encoder.transform([from_activity])[0]
        to_code = self.label_encoder.transform([to_activity])[0]
        
        from_indices = np.where(y == from_code)[0]
        to_indices = np.where(y == to_code)[0]
        
        num_augmentation = int(len(X) * augmentation_ratio)
        
        transition_samples = []
        transition_labels = []
        
        for _ in range(num_augmentation):
            from_idx = np.random.choice(from_indices)
            to_idx = np.random.choice(to_indices)
            
            transition_sample = self.create_transition_sample(
                X[from_idx], X[to_idx], mixing_ratio
            )
            
            transition_samples.append(transition_sample)
            transition_labels.append(from_code)
        
        X_augmented = np.concatenate([X, np.array(transition_samples)], axis=0)
        y_augmented = np.concatenate([y, np.array(transition_labels)], axis=0)
        
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented, num_augmentation
    
    def generate_all_datasets(self, output_dir='./wisdm_datasets', augmentation_ratio=0.10, random_seed=42):
        np.random.seed(random_seed)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        df = self.load_wisdm_raw()
        
        X = df[['x', 'y', 'z']]
        y = df['activity_code']
        
        X_sequences, y_sequences = self.create_dataset(X, y, self.timestep, self.step)
        
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = self.split_data(X_sequences, y_sequences)
        
        original_dir = output_path / 'ORIGINAL'
        original_dir.mkdir(exist_ok=True)
        
        np.save(original_dir / 'X_train.npy', X_train_orig)
        np.save(original_dir / 'y_train.npy', y_train_orig)
        np.save(original_dir / 'X_test.npy', X_test_orig)
        np.save(original_dir / 'y_test.npy', y_test_orig)
        
        dataset_count = 0
        for from_activity, to_activity in self.transitions:
            transition_name = f"{from_activity.upper()}_TO_{to_activity.upper()}"
            
            for mixing_ratio in self.mixing_ratios:
                dataset_count += 1
                ratio_pct = int(mixing_ratio * 100)
                dataset_name = f"{transition_name}_{ratio_pct}pct"
                
                X_aug, y_aug, aug_count = self.create_dataset_with_transition(
                    X_sequences.copy(), y_sequences.copy(),
                    from_activity, to_activity, mixing_ratio, augmentation_ratio
                )
                
                X_train_aug, X_test_aug, y_train_aug, y_test_aug = self.split_data(X_aug, y_aug)
                
                dataset_dir = output_path / dataset_name
                dataset_dir.mkdir(exist_ok=True)
                
                np.save(dataset_dir / 'X_train.npy', X_train_aug)
                np.save(dataset_dir / 'y_train.npy', y_train_aug)
                np.save(dataset_dir / 'X_test.npy', X_test_aug)
                np.save(dataset_dir / 'y_test.npy', y_test_aug)
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python wisdm_transition_generator.py <WISDM_file> [output] [aug_ratio] [seed]")
        print("Example: python wisdm_transition_generator.py WISDM_ar_v1.1_raw.txt ./datasets 0.10 42")
        sys.exit(1)
    
    wisdm_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './wisdm_datasets'
    augmentation_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    if not os.path.exists(wisdm_file):
        print(f"Error: File not found: {wisdm_file}")
        sys.exit(1)
    
    if augmentation_ratio <= 0 or augmentation_ratio >= 1:
        print(f"Error: aug_ratio must be 0 < ratio < 1, got {augmentation_ratio}")
        sys.exit(1)
    
    creator = WISDMTransitionDatasetCreator(wisdm_file)
    creator.generate_all_datasets(output_dir, augmentation_ratio, random_seed)


if __name__ == "__main__":
    main()