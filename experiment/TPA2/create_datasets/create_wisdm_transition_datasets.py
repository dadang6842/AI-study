# -*- coding: utf-8 -*-
"""
WISDM 데이터셋에 전이 구간을 추가하여 데이터셋 생성

시퀀스 분할 방식:
- 앞 클래스의 앞부분 + 뒤 클래스의 뒷부분 연결
- 예: WALKING→JOGGING 10% = WALKING 90% + JOGGING 10%
- 레이블: 앞 클래스
- 시퀀스 분할: timestep=200, step=40
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class WISDMTransitionDatasetCreator:
    def __init__(self, wisdm_data_path):
        """
        Args:
            wisdm_data_path: WISDM raw 데이터 파일 경로 (WISDM_ar_v1.1_raw.txt)
        """
        self.wisdm_data_path = Path(wisdm_data_path)
        
        # LabelEncoder 초기화
        self.label_encoder = LabelEncoder()
        
        # 전이 구간 정의 (from_activity -> to_activity)
        # 원칙: 실제 인간의 자연스러운 행동 전환만 포함
        self.transitions = [
            # === 정적 활동 전이 (2개) ===
            ('Standing', 'Sitting'),   # STAND_TO_SIT (서있다가 앉기)
            ('Sitting', 'Standing'),   # SIT_TO_STAND (앉았다가 일어서기)
            
            # === Standing <-> Walking 전이 (2개) ===
            # 정적 → 저강도 동적 전환
            ('Standing', 'Walking'),   # STAND_TO_WALK (서있다가 걷기 시작)
            ('Walking', 'Standing'),   # WALK_TO_STAND (걷다가 멈춰서기)
            
            # === Walking <-> Jogging 전이 (2개) ===
            # 저강도 → 고강도 동적 전환
            ('Walking', 'Jogging'),    # WALK_TO_JOG (걷다가 뛰기)
            ('Jogging', 'Walking'),    # JOG_TO_WALK (뛰다가 걷기)
            
            # === Walking <-> Stairs 전이 (4개) ===
            # 평지 → 계단 전환 (자주 발생)
            ('Walking', 'Upstairs'),     # WALK_TO_UP (걷다가 계단 올라가기)
            ('Walking', 'Downstairs'),   # WALK_TO_DOWN (걷다가 계단 내려가기)
            ('Upstairs', 'Walking'),     # UP_TO_WALK (계단 올라가다가 평지 걷기)
            ('Downstairs', 'Walking'),   # DOWN_TO_WALK (계단 내려가다가 평지 걷기)
        ]
        
        # 뒤 클래스 분할 비율
        self.mixing_ratios = [0.1, 0.2, 0.3, 0.4]
        
        # 시퀀스 분할 파라미터
        self.timestep = 200
        self.step = 40
        
    def load_wisdm_raw(self):
        """
        WISDM raw 데이터 로드 및 전처리
        
        Returns:
            df: 전처리된 DataFrame
        """
        print("Loading WISDM raw data...")
        
        # 데이터 로드
        column_names = ['user', 'activity', 'timestamp', 'x', 'y', 'z', 'NaN']
        df = pd.read_csv(self.wisdm_data_path, header=None, names=column_names, comment=';')

        df = df.drop('NaN', axis=1).dropna()
        
        # 마지막 컬럼의 세미콜론 제거
        df['z'] = df['z'].replace(regex=True, to_replace=r';', value=r'')

        print(f"x dtype: {df['x'].dtype}, y dtype: {df['y'].dtype}, z dtype: {df['z'].dtype}")
        
        # 결측치 제거
        df = df.dropna()
        
        # LabelEncoder로 활동 레이블을 숫자로 변환
        df['activity_code'] = self.label_encoder.fit_transform(df['activity'])
        
        print(f"Total samples: {len(df)}")
        print(f"Activities: {df['activity'].value_counts().to_dict()}")
        print(f"Activity encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        print(f"Users: {df['user'].nunique()}")
        
        return df
    
    def create_dataset(self, X, y, time_steps=200, step=40):
        """
        슬라이딩 윈도우로 시퀀스 생성
        
        Args:
            X: DataFrame with sensor data (x, y, z columns)
            y: Series with activity labels
            time_steps: 윈도우 크기 (default: 200)
            step: 슬라이딩 스텝 (default: 40)
            
        Returns:
            xs: shape (N, time_steps, 3)
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
            X: shape (N, 200, 3)
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
            from_sample: shape (200, 3) - 앞 클래스 샘플
            to_sample: shape (200, 3) - 뒤 클래스 샘플
            mixing_ratio: 뒤 클래스 분할 비율 (0.1, 0.2, 0.3, 0.4)
            
        Returns:
            transition_sample: shape (200, 3)
            - 앞부분: from_sample의 앞 (1-mixing_ratio) %
            - 뒷부분: to_sample의 뒤 mixing_ratio %
        """
        T, C = from_sample.shape  # (200, 3)
        
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
            X: shape (N, 200, 3)
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
        transition_samples = []
        transition_labels = []
        
        for _ in range(num_augmentation):
            # 랜덤하게 앞/뒤 클래스 샘플 선택
            from_idx = np.random.choice(from_indices)
            to_idx = np.random.choice(to_indices)
            
            # 시퀀스 분할 방식으로 전이 샘플 생성
            transition_sample = self.create_transition_sample(
                X[from_idx], X[to_idx], mixing_ratio
            )
            
            transition_samples.append(transition_sample)
            # 레이블은 앞 클래스
            transition_labels.append(from_code)
        
        # 원본 + 전이 샘플 결합
        X_augmented = np.concatenate([X, np.array(transition_samples)], axis=0)
        y_augmented = np.concatenate([y, np.array(transition_labels)], axis=0)
        
        # 셔플
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented, num_augmentation
    
    def generate_all_datasets(self, output_dir='./wisdm_datasets', augmentation_ratio=0.10):
        """
        모든 데이터셋 생성
        
        Args:
            output_dir: 출력 디렉토리
            augmentation_ratio: 증강 비율 (기본 10%)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("Loading WISDM Raw Data...")
        print("=" * 80)
        
        # WISDM 데이터 로드
        df = self.load_wisdm_raw()
        
        # 센서 데이터와 레이블 분리
        X = df[['x', 'y', 'z']]
        y = df['activity_code']
        
        # 시퀀스 생성
        print(f"\nCreating sequences (timestep={self.timestep}, step={self.step})...")
        X_sequences, y_sequences = self.create_dataset(X, y, self.timestep, self.step)
        print(f"Total sequences created: {len(X_sequences)}")
        print(f"Shape: {X_sequences.shape}")
        
        # 메타데이터 초기화
        metadata = {
            'dataset_name': 'WISDM-HAR',
            'method': 'Sequence Splitting',
            'timestep': self.timestep,
            'step': self.step,
            'augmentation_ratio': augmentation_ratio,
            'train_test_split': '80-20',
            'total_sequences': len(X_sequences),
            'datasets': []
        }
        
        # 원본 데이터셋 저장 (전이 없음, 증강 후 split)
        print("\n" + "=" * 80)
        print("Creating ORIGINAL dataset (no transition)...")
        print("=" * 80)
        
        # 원본 데이터 split
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = self.split_data(X_sequences, y_sequences)
        
        original_dir = output_path / 'ORIGINAL'
        original_dir.mkdir(exist_ok=True)
        
        np.save(original_dir / 'X_train.npy', X_train_orig)
        np.save(original_dir / 'y_train.npy', y_train_orig)
        np.save(original_dir / 'X_test.npy', X_test_orig)
        np.save(original_dir / 'y_test.npy', y_test_orig)
        
        with open(original_dir / 'info.txt', 'w') as f:
            f.write("Dataset: ORIGINAL (No Transition)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total sequences: {len(X_sequences)}\n")
            f.write(f"Train: {len(X_train_orig)} samples\n")
            f.write(f"Test:  {len(X_test_orig)} samples\n")
            f.write(f"Shape: (samples, timestep={self.timestep}, channels=3)\n")
            f.write(f"Sequence params: timestep={self.timestep}, step={self.step}\n")
        
        metadata['datasets'].append({
            'name': 'ORIGINAL',
            'total_size': len(X_sequences),
            'train_size': len(X_train_orig),
            'test_size': len(X_test_orig)
        })
        
        print(f"✓ Total: {len(X_sequences)} sequences")
        print(f"✓ Train: {len(X_train_orig)} samples")
        print(f"✓ Test:  {len(X_test_orig)} samples")
        
        # 전이 데이터셋 생성 (10개 전이 × 4개 비율 = 40개)
        dataset_count = 0
        for from_activity, to_activity in self.transitions:
            transition_name = f"{from_activity.upper()}_TO_{to_activity.upper()}"
            
            for mixing_ratio in self.mixing_ratios:
                dataset_count += 1
                ratio_pct = int(mixing_ratio * 100)
                dataset_name = f"{transition_name}_{ratio_pct}pct"
                
                print("=" * 80)
                print(f"[{dataset_count}/40] Creating: {dataset_name}")
                print("=" * 80)
                print(f"Transition: {from_activity} → {to_activity}")
                print(f"Mixing: Front {100-ratio_pct}% {from_activity} + Back {ratio_pct}% {to_activity}")
                from_label = self.label_encoder.transform([from_activity])[0]
                print(f"Label: {from_activity} ({from_label})")
                
                # 전체 데이터 증강
                X_aug, y_aug, aug_count = self.create_dataset_with_transition(
                    X_sequences.copy(), y_sequences.copy(),
                    from_activity, to_activity, mixing_ratio, augmentation_ratio
                )
                
                # 증강 후 split
                X_train_aug, X_test_aug, y_train_aug, y_test_aug = self.split_data(X_aug, y_aug)
                
                # 저장
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
                    f.write(f"Shape: (samples, timestep={self.timestep}, channels=3)\n")
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
        
        print("\n" + "=" * 80)
        print("✓ All 41 datasets created successfully! (1 original + 40 augmented)")
        print("=" * 80)
        print(f"Output: {output_path.absolute()}")
        print(f"Summary: {output_path / 'summary.txt'}")
        
        return metadata
    
    def _save_summary(self, output_path, metadata):
        """요약 저장"""
        with open(output_path / 'summary.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WISDM TRANSITION DATASETS (Sequence Splitting Method)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total datasets: 41\n")
            f.write(f"  - Original: 1 (no augmentation)\n")
            f.write(f"  - Augmented: 40 (with transitions)\n")
            f.write(f"Method: Sequence Splitting (Time-based)\n")
            f.write(f"Transitions: 10\n")
            f.write(f"  - Static transitions: 2 (STANDING↔SITTING)\n")
            f.write(f"  - Standing-Walking transitions: 2 (STANDING↔WALKING)\n")
            f.write(f"  - Walking-Jogging transitions: 2 (WALKING↔JOGGING)\n")
            f.write(f"  - Walking-Stairs transitions: 4 (WALKING↔UPSTAIRS/DOWNSTAIRS)\n")
            f.write(f"Mixing ratios: 4 (10%, 20%, 30%, 40%)\n")
            f.write(f"Augmentation ratio: {int(metadata['augmentation_ratio']*100)}%\n")
            f.write(f"Sequence params: timestep={self.timestep}, step={self.step}\n")
            f.write(f"Data shape: (samples, timestep={self.timestep}, channels=3)\n")
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
            f.write("   |------ 180 pts ------|  |--- 20 pts --|\n")
            f.write("   Label: WALKING (1)\n\n")
            
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
        print("WISDM Transition Dataset Generator (Sequence Splitting)")
        print("=" * 80)
        print("\nUsage:")
        print("  python create_wisdm_transition_datasets.py <WISDM_file> [output] [aug_ratio]")
        print("\nArguments:")
        print("  WISDM_file   : WISDM raw data file path (required)")
        print("                 e.g., WISDM_ar_v1.1_raw.txt")
        print("  output       : Output directory (default: ./wisdm_datasets)")
        print("  aug_ratio    : Augmentation ratio (default: 0.10)")
        print("\nExamples:")
        print("  python create_wisdm_transition_datasets.py WISDM_ar_v1.1_raw.txt")
        print("  python create_wisdm_transition_datasets.py WISDM_ar_v1.1_raw.txt ./datasets 0.15")
        print("\nSequence Parameters:")
        print("  - timestep: 200")
        print("  - step: 40")
        print("\nData Split:")
        print("  - Augmentation FIRST, then split")
        print("  - Train: 80%")
        print("  - Test:  20%")
        print("  - (Val split will be done during training)")
        print("\nTransitions (10 types):")
        print("  Static: STANDING↔SITTING (2 transitions)")
        print("  Standing-Walking: STANDING↔WALKING (2 transitions)")
        print("  Walking-Jogging: WALKING↔JOGGING (2 transitions)")
        print("  Walking-Stairs: WALKING↔UPSTAIRS/DOWNSTAIRS (4 transitions)")
        print("\nOutput:")
        print("  - 41 datasets total")
        print("    * 1 original (no transition)")
        print("    * 40 augmented (10 transitions × 4 split ratios)")
        print("\nMethod:")
        print("  - Sequence splitting (time-based)")
        print("  - Front X% + Back (100-X)%")
        print("  - Label: Front class")
        print("=" * 80)
        sys.exit(1)
    
    wisdm_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './wisdm_datasets'
    augmentation_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    
    if not os.path.exists(wisdm_file):
        print(f"❌ Error: File not found: {wisdm_file}")
        sys.exit(1)
    
    if augmentation_ratio <= 0 or augmentation_ratio >= 1:
        print(f"❌ Error: aug_ratio must be 0 < ratio < 1, got {augmentation_ratio}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"WISDM file: {wisdm_file}")
    print(f"Output: {output_dir}")
    print(f"Augmentation: {int(augmentation_ratio*100)}%")
    print(f"Method: Sequence Splitting")
    print(f"Sequence params: timestep=200, step=40")
    print(f"Data split: Augment FIRST → then Train(80%) / Test(20%)")
    print("=" * 80 + "\n")
    
    creator = WISDMTransitionDatasetCreator(wisdm_file)
    metadata = creator.generate_all_datasets(output_dir, augmentation_ratio)
    
    print("\n✓ Completed!\n")


if __name__ == "__main__":
    main()
