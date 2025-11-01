# -*- coding: utf-8 -*-
"""
mHealth 데이터셋에 전이 구간을 추가하여 데이터셋 생성

시퀀스 분할 방식:
- 앞 클래스의 앞부분 + 뒤 클래스의 뒷부분 연결
- 예: WALKING→JOGGING 10% = WALKING 90% + JOGGING 10%
- 레이블: 앞 클래스
- 시퀀스 분할: timestep=200, step=100
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class MHealthTransitionDatasetCreator:
    def __init__(self, mhealth_data_dir):
        """
        Args:
            mhealth_data_dir: mHealth 데이터 디렉토리 경로
        """
        self.mhealth_data_dir = Path(mhealth_data_dir)
        
        # LabelEncoder 초기화
        self.label_encoder = LabelEncoder()
        
        # 활동 이름 매핑 (0-based index)
        # 원본 label 1~12가 0~11로 변환됨
        self.activity_names = {
            0: 'Standing',
            1: 'Sitting',
            2: 'Lying',
            3: 'Walking',
            4: 'Climbing_stairs',
            5: 'Waist_bends',
            6: 'Frontal_elevation',
            7: 'Knees_bending',
            8: 'Cycling',
            9: 'Jogging',
            10: 'Running',
            11: 'Jump'
        }
        
        # 전이 구간 정의 (from_activity -> to_activity)
        # 원칙: 실제 인간의 자연스러운 행동 전환만 포함
        # Note: WISDM의 Jogging과 대응하여 Jogging 사용 (Running 대신)
        self.transitions = [
            # === 정적 활동 전이 (6개) ===
            ('Standing', 'Sitting'),     # 서있다가 앉기
            ('Sitting', 'Standing'),     # 앉았다가 일어서기
            ('Sitting', 'Lying'),        # 앉았다가 눕기
            ('Lying', 'Sitting'),        # 누워있다가 앉기
            ('Standing', 'Lying'),       # 서있다가 눕기
            ('Lying', 'Standing'),       # 누워있다가 일어서기
            
            # === Standing <-> Walking 전이 (2개) ===
            ('Standing', 'Walking'),     # 서있다가 걷기 시작
            ('Walking', 'Standing'),     # 걷다가 멈춰서기
            
            # === Walking <-> Jogging 전이 (2개) ===
            # WISDM의 Jogging과 대응
            ('Walking', 'Jogging'),      # 걷다가 조깅
            ('Jogging', 'Walking'),      # 조깅하다가 걷기
            
            # === Walking <-> Stairs 전이 (2개) ===
            ('Walking', 'Climbing_stairs'),    # 걷다가 계단 오르기
            ('Climbing_stairs', 'Walking'),    # 계단 오르다가 평지 걷기
        ]
        
        # 뒤 클래스 분할 비율
        self.mixing_ratios = [0.1, 0.2, 0.3, 0.4]
        
        # 시퀀스 분할 파라미터
        self.timestep = 200
        self.step = 100
        
    def load_mhealth_data(self):
        """
        mHealth 데이터 로드 및 전처리
        
        Returns:
            df: 전처리된 DataFrame
        """
        print("Loading mHealth data...")
        
        # 컬럼명 정의 (23개 센서)
        col_names = [
            'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
            'ecg1', 'ecg2',
            'acc_left_ankle_x', 'acc_left_ankle_y', 'acc_left_ankle_z',
            'gyro_left_ankle_x', 'gyro_left_ankle_y', 'gyro_left_ankle_z',
            'magnet_left_ankle_x', 'magnet_left_ankle_y', 'magnet_left_ankle_z',
            'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z',
            'gyro_right_arm_x', 'gyro_right_arm_y', 'gyro_right_arm_z',
            'magnet_right_arm_x', 'magnet_right_arm_y', 'magnet_right_arm_z',
            'label'
        ]
        
        # 모든 subject 파일 로드 (subject 1~10)
        dfs = []
        for i in range(1, 11):
            path = self.mhealth_data_dir / f"mHealth_subject{i}.log"
            print(f"  Loading {path.name}...")
            df_i = pd.read_csv(path, sep='\t', header=None)
            dfs.append(df_i)
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna(how='any', axis=0)
        df.columns = col_names
        
        # Remove label 0 (null/no activity)
        df = df[df.label != 0]
        
        # Adjust labels to start from 0 (1->0, 2->1, ..., 12->11)
        df['label'] = df['label'] - 1
        
        # Map activity codes to names
        df['activity_name'] = df['label'].map(self.activity_names)
        
        # Fit label encoder
        self.label_encoder.fit(list(self.activity_names.values()))
        
        print(f"Processed data shape: {df.shape}")
        print(f"Activities: {df['activity_name'].value_counts().to_dict()}")
        print(f"Activity encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df
    
    def create_dataset(self, X, y, time_steps=200, step=100):
        """
        슬라이딩 윈도우로 시퀀스 생성
        
        Args:
            X: DataFrame with sensor data (23 features)
            y: Series with activity labels
            time_steps: 윈도우 크기 (default: 100)
            step: 슬라이딩 스텝 (default: 50)
            
        Returns:
            xs: shape (N, time_steps, 23)
            ys: shape (N,)
        """
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
        """
        데이터를 train/test로 분할
        
        Args:
            X: shape (N, 100, 23)
            y: shape (N,)
            test_size: test 비율 (default: 0.2)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def create_transition_sample(self, from_sample, to_sample, mixing_ratio):
        """
        시퀀스 분할 방식으로 전이 샘플 생성
        
        Args:
            from_sample: shape (100, 23) - 앞 클래스 샘플
            to_sample: shape (100, 23) - 뒤 클래스 샘플
            mixing_ratio: 뒤 클래스 분할 비율 (0.1, 0.2, 0.3, 0.4)
            
        Returns:
            transition_sample: shape (100, 23)
            - 앞부분: from_sample의 앞 (1-mixing_ratio) %
            - 뒷부분: to_sample의 뒤 mixing_ratio %
        """
        T, C = from_sample.shape  # (100, 23)
        
        # 분할 지점 계산
        split_point = int(T * (1 - mixing_ratio))
        
        # 전이 샘플 생성
        transition_sample = np.zeros_like(from_sample)
        transition_sample[:split_point, :] = from_sample[:split_point, :]
        transition_sample[split_point:, :] = to_sample[:T - split_point, :]
        
        return transition_sample
    
    def create_dataset_with_transition(self, X, y, from_activity, to_activity,
                                       mixing_ratio, augmentation_ratio):
        """
        특정 전이 구간으로 데이터셋 증강
        
        Args:
            X: shape (N, 100, 23)
            y: shape (N,)
            from_activity: 앞 클래스
            to_activity: 뒤 클래스
            mixing_ratio: 뒤 클래스 분할 비율 (10%, 20%, 30%, 40%)
            augmentation_ratio: 증강 비율 (전체 데이터 대비)
            
        Returns:
            X_augmented, y_augmented, augmentation_count
        """
        # LabelEncoder로 활동명을 코드로 변환
        from_code = self.label_encoder.transform([from_activity])[0]
        to_code = self.label_encoder.transform([to_activity])[0]
        
        # 해당 활동의 샘플 인덱스
        from_indices = np.where(y == from_code)[0]
        to_indices = np.where(y == to_code)[0]
        
        # 증강할 샘플 수
        num_augmentation = int(len(X) * augmentation_ratio)
        
        # 전이 샘플 생성
        X_transition = []
        y_transition = []
        
        for _ in range(num_augmentation):
            # 랜덤하게 샘플 선택
            from_idx = np.random.choice(from_indices)
            to_idx = np.random.choice(to_indices)
            
            # 전이 샘플 생성
            transition_sample = self.create_transition_sample(
                X[from_idx], X[to_idx], mixing_ratio
            )
            
            X_transition.append(transition_sample)
            y_transition.append(from_code)  # 레이블은 앞 클래스
        
        X_transition = np.array(X_transition)
        y_transition = np.array(y_transition)
        
        # 원본과 증강 데이터 합치기
        X_augmented = np.concatenate([X, X_transition], axis=0)
        y_augmented = np.concatenate([y, y_transition], axis=0)
        
        # 셔플
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented, num_augmentation
    
    def generate_all_datasets(self, output_dir, augmentation_ratio=0.10):
        """
        모든 전이 조합으로 데이터셋 생성
        
        Args:
            output_dir: 출력 디렉토리
            augmentation_ratio: 증강 비율 (default: 0.10)
            
        Returns:
            metadata: 메타데이터
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # mHealth 데이터 로드 및 전처리
        df = self.load_mhealth_data()
        
        # Feature와 label 분리
        X = df.drop(columns=['label', 'activity_name'])
        y = df['label']
        
        # 시퀀스 생성
        print(f"\nCreating sequences (timestep={self.timestep}, step={self.step})...")
        X_sequences, y_sequences = self.create_dataset(X, y, self.timestep, self.step)
        
        print(f"Sequences shape: {X_sequences.shape}")
        print(f"Labels shape: {y_sequences.shape}")
        
        # 메타데이터 초기화
        metadata = {
            'timestep': self.timestep,
            'step': self.step,
            'num_features': 23,
            'num_classes': 12,
            'augmentation_ratio': augmentation_ratio,
            'activity_names': self.activity_names,
            'datasets': []
        }
        
        # 1. 원본 데이터셋 (전이 없음)
        print("\n" + "=" * 80)
        print("Creating ORIGINAL dataset (no transition)...")
        print("=" * 80)
        
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = self.split_data(
            X_sequences, y_sequences
        )
        
        # 원본 데이터셋 저장
        original_dir = output_path / 'ORIGINAL'
        original_dir.mkdir(exist_ok=True)
        
        np.save(original_dir / 'X_train.npy', X_train_orig)
        np.save(original_dir / 'y_train.npy', y_train_orig)
        np.save(original_dir / 'X_test.npy', X_test_orig)
        np.save(original_dir / 'y_test.npy', y_test_orig)
        
        with open(original_dir / 'info.txt', 'w') as f:
            f.write("Dataset: ORIGINAL\n")
            f.write("=" * 60 + "\n\n")
            f.write("Original dataset (no transition)\n")
            f.write(f"Total: {len(X_sequences)} sequences\n")
            f.write(f"Train: {len(X_train_orig)} samples\n")
            f.write(f"Test:  {len(X_test_orig)} samples\n")
            f.write(f"Shape: (samples, timestep={self.timestep}, channels=23)\n")
            f.write(f"Sequence params: timestep={self.timestep}, step={self.step}\n")
        
        metadata['datasets'].append({
            'name': 'ORIGINAL',
            'transition': None,
            'from_activity': None,
            'to_activity': None,
            'mixing_ratio': None,
            'augmentation_ratio': 0,
            'original_size': len(X_sequences),
            'augmented_size': len(X_sequences),
            'augmentation_count': 0,
            'train_size': len(X_train_orig),
            'test_size': len(X_test_orig)
        })
        
        print(f"\n✓ Train: {len(X_train_orig)} samples")
        print(f"✓ Test:  {len(X_test_orig)} samples")
        print(f"✓ Saved to: {original_dir}\n")
        
        # 2. 전이 데이터셋들 (12 transitions × 4 ratios = 48 datasets)
        for from_activity, to_activity in self.transitions:
            for mixing_ratio in self.mixing_ratios:
                ratio_pct = int(mixing_ratio * 100)
                transition_name = f"{from_activity}_TO_{to_activity}_{ratio_pct}PCT"
                dataset_name = transition_name
                
                print("\n" + "=" * 80)
                print(f"Creating {dataset_name}...")
                print("=" * 80)
                print(f"Transition: {from_activity} → {to_activity}")
                print(f"Composition: Front {100-ratio_pct}% {from_activity} + Back {ratio_pct}% {to_activity}")
                print(f"Label: {from_activity}")
                print(f"Augmentation: {int(augmentation_ratio*100)}%")
                
                # 데이터 증강
                X_aug, y_aug, aug_count = self.create_dataset_with_transition(
                    X_sequences, y_sequences, from_activity, to_activity,
                    mixing_ratio, augmentation_ratio
                )
                
                # Train/Test 분할 (증강 후 분할)
                X_train_aug, X_test_aug, y_train_aug, y_test_aug = self.split_data(
                    X_aug, y_aug
                )
                
                # 데이터셋 저장
                dataset_dir = output_path / dataset_name
                dataset_dir.mkdir(exist_ok=True)
                
                np.save(dataset_dir / 'X_train.npy', X_train_aug)
                np.save(dataset_dir / 'y_train.npy', y_train_aug)
                np.save(dataset_dir / 'X_test.npy', X_test_aug)
                np.save(dataset_dir / 'y_test.npy', y_test_aug)
                
                # 정보 저장
                dataset_info = {
                    'name': dataset_name,
                    'transition': transition_name,
                    'from_activity': from_activity,
                    'to_activity': to_activity,
                    'from_label': self.label_encoder.transform([from_activity])[0],
                    'mixing_ratio': mixing_ratio,
                    'augmentation_ratio': augmentation_ratio,
                    'original_size': len(X_sequences),
                    'augmented_size': len(X_aug),
                    'augmentation_count': aug_count,
                    'train_size': len(X_train_aug),
                    'test_size': len(X_test_aug)
                }
                
                with open(dataset_dir / 'info.txt', 'w') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Method: Sequence Splitting\n")
                    f.write(f"Transition: {from_activity} → {to_activity}\n")
                    f.write(f"Composition: Front {100-ratio_pct}% {from_activity} + Back {ratio_pct}% {to_activity}\n")
                    f.write(f"Label: {from_activity} (code: {dataset_info['from_label']})\n")
                    f.write(f"Augmentation: {int(augmentation_ratio*100)}%\n\n")
                    f.write(f"Original: {len(X_sequences)} sequences\n")
                    f.write(f"Augmented: {len(X_aug)} sequences (+{aug_count})\n")
                    f.write(f"Train: {len(X_train_aug)} samples\n")
                    f.write(f"Test:  {len(X_test_aug)} samples\n")
                    f.write(f"Shape: (samples, timestep={self.timestep}, channels=23)\n")
                    f.write(f"Sequence params: timestep={self.timestep}, step={self.step}\n")
                
                metadata['datasets'].append(dataset_info)
                
                print(f"\n✓ Original: {len(X_sequences)} → Augmented: {len(X_aug)} (+{aug_count})")
                print(f"✓ Train: {len(X_train_aug)} samples")
                print(f"✓ Test:  {len(X_test_aug)} samples")
                print(f"✓ Saved to: {dataset_dir}\n")
        
        # 메타데이터 저장
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # 요약 저장
        self._save_summary(output_path, metadata)
        
        total_datasets = len(metadata['datasets'])
        print("\n" + "=" * 80)
        print(f"✓ All {total_datasets} datasets created successfully!")
        print("=" * 80)
        print(f"Output: {output_path.absolute()}")
        print(f"Summary: {output_path / 'summary.txt'}")
        
        return metadata
    
    def _save_summary(self, output_path, metadata):
        """요약 저장"""
        num_datasets = len(metadata['datasets'])
        num_augmented = num_datasets - 1
        
        with open(output_path / 'summary.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("mHealth TRANSITION DATASETS (Sequence Splitting Method)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total datasets: {num_datasets}\n")
            f.write(f"  - Original: 1 (no augmentation)\n")
            f.write(f"  - Augmented: {num_augmented} (with transitions)\n")
            f.write(f"Method: Sequence Splitting (Time-based)\n")
            f.write(f"Transitions: 12\n")
            f.write(f"  - Static transitions: 6 (STANDING↔SITTING↔LYING)\n")
            f.write(f"  - Standing-Walking transitions: 2 (STANDING↔WALKING)\n")
            f.write(f"  - Walking-Jogging transitions: 2 (WALKING↔JOGGING)\n")
            f.write(f"  - Walking-Stairs transitions: 2 (WALKING↔CLIMBING_STAIRS)\n")
            f.write(f"  Note: Using Jogging (not Running) to match WISDM\n")
            f.write(f"Mixing ratios: 4 (10%, 20%, 30%, 40%)\n")
            f.write(f"Augmentation ratio: {int(metadata['augmentation_ratio']*100)}%\n")
            f.write(f"Sequence params: timestep={self.timestep}, step={self.step}\n")
            f.write(f"Data shape: (samples, timestep={self.timestep}, channels=23)\n")
            f.write(f"Data split: Train/Test = 80/20 (after augmentation)\n")
            f.write(f"Labeling: Front class\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("METHOD EXPLANATION\n")
            f.write("=" * 80 + "\n")
            f.write("Sequence Splitting:\n")
            f.write("  - 10%: Front 90% from class A + Back 10% from class B\n")
            f.write("  - 20%: Front 80% from class A + Back 20% from class B\n")
            f.write("  - 30%: Front 70% from class A + Back 30% from class B\n")
            f.write("  - 40%: Front 60% from class A + Back 40% from class B\n")
            f.write("  - Label: Always class A (front class)\n\n")
            
            f.write("Example (WALKING→JOGGING 10%):\n")
            f.write("  [WALK WALK ... WALK | JOG JOG ... JOG]\n")
            f.write("   |------ 90 pts ------|  |--- 10 pts --|\n")
            f.write("   Label: WALKING (3)\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DATASET LIST\n")
            f.write("=" * 80 + "\n\n")
            
            for i, ds in enumerate(metadata['datasets'], 1):
                f.write(f"{i:2d}. {ds['name']}\n")
                
                # 원본 데이터셋 (전이 없음)인 경우
                if ds['name'] == 'ORIGINAL':
                    f.write(f"    Original dataset (no transition)\n")
                    f.write(f"    Train: {ds['train_size']} samples\n")
                    f.write(f"    Test:  {ds['test_size']} samples\n\n")
                else:
                    # 전이 데이터셋인 경우
                    f.write(f"    {ds['from_activity']} → {ds['to_activity']}\n")
                    f.write(f"    Mix: Front {100-int(ds['mixing_ratio']*100)}% + Back {int(ds['mixing_ratio']*100)}%\n")
                    f.write(f"    Label: {ds['from_activity']} ({ds['from_label']})\n")
                    f.write(f"    Original: {ds['original_size']} → Augmented: {ds['augmented_size']} "
                           f"(+{ds['augmentation_count']})\n")
                    f.write(f"    Train: {ds['train_size']} samples\n")
                    f.write(f"    Test:  {ds['test_size']} samples\n\n")


def main():
    """메인 실행"""
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 80)
        print("mHealth Transition Dataset Generator (Sequence Splitting)")
        print("=" * 80)
        print("\nUsage:")
        print("  python create_mhealth_transition_datasets.py <mHealth_dir> [output] [aug_ratio]")
        print("\nArguments:")
        print("  mHealth_dir  : mHealth data directory path (required)")
        print("                 e.g., ./mHealth_dataset")
        print("  output       : Output directory (default: ./mhealth_datasets)")
        print("  aug_ratio    : Augmentation ratio (default: 0.10)")
        print("\nExamples:")
        print("  python create_mhealth_transition_datasets.py ./mHealth_dataset")
        print("  python create_mhealth_transition_datasets.py ./mHealth_dataset ./datasets 0.15")
        print("\nSequence Parameters:")
        print("  - timestep: 200")
        print("  - step: 100")
        print("\nData Split:")
        print("  - Augmentation FIRST, then split")
        print("  - Train: 80%")
        print("  - Test:  20%")
        print("  - (Val split will be done during training)")
        print("\nTransitions (12 types):")
        print("  Static: STANDING↔SITTING↔LYING (6 transitions)")
        print("  Standing-Walking: STANDING↔WALKING (2 transitions)")
        print("  Walking-Jogging: WALKING↔JOGGING (2 transitions)")
        print("    Note: Using Jogging (not Running) to match WISDM")
        print("  Walking-Stairs: WALKING↔CLIMBING_STAIRS (2 transitions)")
        print("\nOutput:")
        print("  - 49 datasets total")
        print("    * 1 original (no transition)")
        print("    * 48 augmented (12 transitions × 4 split ratios)")
        print("\nMethod:")
        print("  - Sequence splitting (time-based)")
        print("  - Front X% + Back (100-X)%")
        print("  - Label: Front class")
        print("=" * 80)
        sys.exit(1)
    
    mhealth_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './mhealth_datasets'
    augmentation_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    
    if not os.path.exists(mhealth_dir):
        print(f"❌ Error: Directory not found: {mhealth_dir}")
        sys.exit(1)
    
    if augmentation_ratio <= 0 or augmentation_ratio >= 1:
        print(f"❌ Error: aug_ratio must be 0 < ratio < 1, got {augmentation_ratio}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"mHealth dir: {mhealth_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentation: {int(augmentation_ratio*100)}%")
    print(f"Method: Sequence Splitting")
    print(f"Sequence params: timestep=200, step=100")
    print(f"Data split: Augment FIRST → then Train(80%) / Test(20%)")
    print(f"Note: Using Jogging (not Running) to match WISDM")
    print("=" * 80 + "\n")
    
    creator = MHealthTransitionDatasetCreator(mhealth_dir)
    metadata = creator.generate_all_datasets(output_dir, augmentation_ratio)
    
    print("\n✓ Completed!\n")


if __name__ == "__main__":
    main()
