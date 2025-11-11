import numpy as np
import os
from pathlib import Path
import sys

class TransitionDatasetCreator:
    def __init__(self, uci_data_path):
        self.uci_data_path = Path(uci_data_path)

        self.activity_labels = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS',
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }

        self.transitions = [
            ('STANDING', 'SITTING'),
            ('SITTING', 'STANDING'),
            ('SITTING', 'LAYING'),
            ('LAYING', 'SITTING'),
            ('STANDING', 'LAYING'),
            ('LAYING', 'STANDING'),
            ('STANDING', 'WALKING'),
            ('WALKING', 'STANDING'),
            ('WALKING', 'WALKING_UPSTAIRS'),
            ('WALKING', 'WALKING_DOWNSTAIRS'),
            ('WALKING_UPSTAIRS', 'WALKING'),
            ('WALKING_DOWNSTAIRS', 'WALKING')
        ]

        self.mixing_ratios = [0.1, 0.2, 0.3, 0.4]

        self.signal_files = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

    def load_inertial_signals(self, split='train'):
        base_path = self.uci_data_path / split / 'Inertial Signals'

        signals = []
        for signal_name in self.signal_files:
            file_path = base_path / f'{signal_name}_{split}.txt'
            signal_data = np.loadtxt(file_path)
            signals.append(signal_data)

        X = np.stack(signals, axis=1)

        label_path = self.uci_data_path / split / f'y_{split}.txt'
        y = np.loadtxt(label_path, dtype=int)

        return X, y

    def get_activity_code(self, activity_name):
        for code, name in self.activity_labels.items():
            if name == activity_name:
                return code
        return None

    def create_transition_sample(self, from_sample, to_sample, mixing_ratio):
        C, T = from_sample.shape

        split_point = int(T * (1 - mixing_ratio))

        transition_sample = np.zeros_like(from_sample)
        transition_sample[:, :split_point] = from_sample[:, :split_point]
        transition_sample[:, split_point:] = to_sample[:, :T - split_point]

        return transition_sample

    def create_dataset_with_transition(self, X, y, from_activity, to_activity,
                                       mixing_ratio, augmentation_ratio):
        from_code = self.get_activity_code(from_activity)
        to_code = self.get_activity_code(to_activity)

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

    def generate_all_datasets(self, output_dir='./uci_datasets', augmentation_ratio=0.10, random_seed=42):
        np.random.seed(random_seed)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        X_train, y_train = self.load_inertial_signals('train')
        X_test, y_test = self.load_inertial_signals('test')

        original_dir = output_path / 'ORIGINAL'
        original_dir.mkdir(exist_ok=True)
        np.save(original_dir / 'X_train.npy', X_train)
        np.save(original_dir / 'y_train.npy', y_train)
        np.save(original_dir / 'X_test.npy', X_test)
        np.save(original_dir / 'y_test.npy', y_test)

        for from_activity, to_activity in self.transitions:
            transition_name = f"{from_activity}_TO_{to_activity}"

            for mixing_ratio in self.mixing_ratios:
                ratio_pct = int(mixing_ratio * 100)
                dataset_name = f"{transition_name}_{ratio_pct}pct"

                X_train_aug, y_train_aug, _ = self.create_dataset_with_transition(
                    X_train.copy(), y_train.copy(),
                    from_activity, to_activity, mixing_ratio, augmentation_ratio
                )

                X_test_aug, y_test_aug, _ = self.create_dataset_with_transition(
                    X_test.copy(), y_test.copy(),
                    from_activity, to_activity, mixing_ratio, augmentation_ratio
                )

                dataset_dir = output_path / dataset_name
                dataset_dir.mkdir(exist_ok=True)

                np.save(dataset_dir / 'X_train.npy', X_train_aug)
                np.save(dataset_dir / 'y_train.npy', y_train_aug)
                np.save(dataset_dir / 'X_test.npy', X_test_aug)
                np.save(dataset_dir / 'y_test.npy', y_test_aug)


def main():
    if len(sys.argv) < 2:
        print("Usage: python uci-har_transition_generator.py <UCI_HAR_path> [output] [aug_ratio] [seed]")
        print("Example: python uci-har_transition_generator.py ./UCI_HAR_Dataset ./output 0.10 42")
        sys.exit(1)

    uci_data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './uci_datasets'
    augmentation_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    if not os.path.exists(uci_data_path):
        print(f"Error: Path not found: {uci_data_path}")
        sys.exit(1)

    if augmentation_ratio <= 0 or augmentation_ratio >= 1:
        print(f"Error: aug_ratio must be 0 < ratio < 1, got {augmentation_ratio}")
        sys.exit(1)

    creator = TransitionDatasetCreator(uci_data_path)
    creator.generate_all_datasets(output_dir, augmentation_ratio, random_seed)


if __name__ == "__main__":
    main()