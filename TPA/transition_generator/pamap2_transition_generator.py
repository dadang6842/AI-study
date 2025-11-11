import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys    

class PAMAP2TransitionDatasetCreator:
    def __init__(self, pamap2_csv_path):
        self.pamap2_csv_path = Path(pamap2_csv_path)
        
        self.label_encoder = LabelEncoder()
        
        self.activity_names = {
            0: 'Lying',
            1: 'Sitting',
            2: 'Standing',
            3: 'Walking',
            4: 'Running',
            5: 'Cycling',
            6: 'Nordic_walking',
            7: 'Ascending_stairs',
            8: 'Descending_stairs',
            9: 'Vacuum_cleaning',
            10: 'Ironing',
            11: 'Rope_jumping'
        }
        
        self.transitions = [
            ('Standing', 'Sitting'),
            ('Sitting', 'Standing'),
            ('Sitting', 'Lying'),
            ('Lying', 'Sitting'),
            ('Standing', 'Lying'),
            ('Lying', 'Standing'),
            
            ('Standing', 'Walking'),
            ('Walking', 'Standing'),
            
            ('Walking', 'Running'),
            ('Running', 'Walking'),
            
            ('Walking', 'Ascending_stairs'),
            ('Walking', 'Descending_stairs'),
            ('Ascending_stairs', 'Walking'),
            ('Descending_stairs', 'Walking'),
        ]
        
        self.mixing_ratios = [0.1, 0.2, 0.3, 0.4]
        
        self.timestep = 100
        self.step = 50
        
    def load_pamap2_data(self):
        df = pd.read_csv(self.pamap2_csv_path)
        
        columns_to_drop = [
            'timestamp', 'heart_rate',
            'hand_temp', 'hand_acc_6g_x', 'hand_acc_6g_y', 'hand_acc_6g_z',
            'hand_orient_1', 'hand_orient_2', 'hand_orient_3', 'hand_orient_4',
            'chest_temp', 'chest_acc_6g_x', 'chest_acc_6g_y', 'chest_acc_6g_z',
            'chest_orient_1', 'chest_orient_2', 'chest_orient_3', 'chest_orient_4',
            'ankle_temp', 'ankle_acc_6g_x', 'ankle_acc_6g_y', 'ankle_acc_6g_z',
            'ankle_orient_1', 'ankle_orient_2', 'ankle_orient_3', 'ankle_orient_4', 'subject'
        ]
        df = df.drop(columns=columns_to_drop)
        
        activity_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
            12: 7, 13: 8, 16: 9, 17: 10, 24: 11
        }
        df = df[df['activityID'].isin(activity_mapping.keys())]
        
        df_list = []
        for activity_id in df['activityID'].unique():
            activity_df = df[df['activityID'] == activity_id].copy()
            numeric_cols = activity_df.select_dtypes(exclude='object').columns
            activity_df[numeric_cols] = activity_df[numeric_cols].interpolate(method='linear')
            activity_df[numeric_cols] = activity_df[numeric_cols].ffill().bfill()
            df_list.append(activity_df)
        
        df_processed = pd.concat(df_list, ignore_index=True)
        
        df_processed['activity_code'] = df_processed['activityID'].map(activity_mapping)
        df_processed['activity_name'] = df_processed['activity_code'].map(self.activity_names)
        
        self.label_encoder.fit(list(self.activity_names.values()))
        
        return df_processed
    
    def create_dataset(self, X, y, time_steps=100, step=50):
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
        
        X_transition = []
        y_transition = []
        
        for _ in range(num_augmentation):
            from_idx = np.random.choice(from_indices)
            to_idx = np.random.choice(to_indices)
            
            transition_sample = self.create_transition_sample(
                X[from_idx], X[to_idx], mixing_ratio
            )
            
            X_transition.append(transition_sample)
            y_transition.append(from_code)
        
        X_transition = np.array(X_transition)
        y_transition = np.array(y_transition)
        
        X_augmented = np.concatenate([X, X_transition], axis=0)
        y_augmented = np.concatenate([y, y_transition], axis=0)
        
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented, num_augmentation
    
    def generate_all_datasets(self, output_dir, augmentation_ratio=0.10, random_seed=42):
        np.random.seed(random_seed)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = self.load_pamap2_data()
        
        X = df.drop(columns=['activityID', 'activity_code', 'activity_name'])
        y = df['activity_code']
        
        X_sequences, y_sequences = self.create_dataset(X, y, self.timestep, self.step)
        
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = self.split_data(
            X_sequences, y_sequences
        )
        
        original_dir = output_path / 'ORIGINAL'
        original_dir.mkdir(exist_ok=True)
        
        np.save(original_dir / 'X_train.npy', X_train_orig.astype(np.float32))
        np.save(original_dir / 'y_train.npy', y_train_orig.astype(np.int64))
        np.save(original_dir / 'X_test.npy', X_test_orig.astype(np.float32))
        np.save(original_dir / 'y_test.npy', y_test_orig.astype(np.int64))
        
        for from_activity, to_activity in self.transitions:
            for mixing_ratio in self.mixing_ratios:
                ratio_pct = int(mixing_ratio * 100)
                transition_name = f"{from_activity}_TO_{to_activity}_{ratio_pct}PCT"
                dataset_name = transition_name
                
                X_aug, y_aug, aug_count = self.create_dataset_with_transition(
                    X_sequences, y_sequences, from_activity, to_activity,
                    mixing_ratio, augmentation_ratio
                )
                
                X_train_aug, X_test_aug, y_train_aug, y_test_aug = self.split_data(
                    X_aug, y_aug
                )
                
                dataset_dir = output_path / dataset_name
                dataset_dir.mkdir(exist_ok=True)
                
                np.save(dataset_dir / 'X_train.npy', X_train_aug.astype(np.float32))
                np.save(dataset_dir / 'y_train.npy', y_train_aug.astype(np.int64))
                np.save(dataset_dir / 'X_test.npy', X_test_aug.astype(np.float32))
                np.save(dataset_dir / 'y_test.npy', y_test_aug.astype(np.int64))
    
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python pamap2_transition_generator.py <PAMAP2_csv> [output] [aug_ratio] [seed]")
        print("Example: python pamap2_transition_generator.py pamap2_data.csv ./datasets 0.10 42")
        sys.exit(1)
    
    pamap2_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './pamap2_datasets'
    augmentation_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    if not os.path.exists(pamap2_csv):
        print(f"Error: File not found: {pamap2_csv}")
        sys.exit(1)
    
    if augmentation_ratio <= 0 or augmentation_ratio >= 1:
        print(f"Error: aug_ratio must be 0 < ratio < 1, got {augmentation_ratio}")
        sys.exit(1)
    
    creator = PAMAP2TransitionDatasetCreator(pamap2_csv)
    creator.generate_all_datasets(output_dir, augmentation_ratio, random_seed)


if __name__ == "__main__":
    main()