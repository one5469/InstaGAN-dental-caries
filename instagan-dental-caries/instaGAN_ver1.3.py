"""
InstaGAN for Dental Caries Generation - 평가 메트릭 추가 버전
FID, IS, LPIPS를 통한 치아 위치별 생성 품질 평가
"""
import warnings

# 파일 맨 위에 추가
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
from scipy import linalg
import lpips

# ============================================
# 설정
# ============================================
GPU_ID = 3
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DIR = os.environ.get("INSTAGAN_DATA_DIR", "./data/teeth_analysis")
METADATA_FILE = f"{DATA_DIR}/metadata_v20_1_20251123_Validation_data.json"
LABEL_DIR_LOWER = f"{DATA_DIR}/Validation Data/라벨데이터/lower/5.lower"
LABEL_DIR_UPPER = f"{DATA_DIR}/Validation Data/라벨데이터/upper/4.upper"
IMAGE_DIR_LOWER = f"{DATA_DIR}/Validation Data/원데이터/lower/5.lower"
IMAGE_DIR_UPPER = f"{DATA_DIR}/Validation Data/원데이터/upper/4.upper"

OUTPUT_DIR = os.environ.get("INSTAGAN_OUTPUT_DIR", "./outputs")
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
DEBUG_DIR = f"{OUTPUT_DIR}/debug"
METRICS_DIR = f"{OUTPUT_DIR}/metrics"  # 메트릭 결과 저장 디렉토리
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TOOTH_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 500
LR_G = 0.0002
LR_D = 0.0002

# ============================================
# 평가 메트릭 클래스
# ============================================
class InceptionScoreCalculator:
    """Inception Score 계산"""
    def __init__(self, device='cuda', resize=True):
        self.device = device
        self.resize = resize
        
        # weights=Inception_V3_Weights.IMAGENET1K_V1
        self.inception_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, 
            transform_input=False
        ).to(device)
        self.inception_model.eval()
        
        if resize:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def calculate_inception_score(self, images, splits=50):
        """
        Inception Score 계산
        Args:
            images: (N, 3, H, W) tensor, 값 범위 [0, 1]
            splits: IS 계산시 분할 수
        Returns:
            mean_is: Inception Score 평균
            std_is: Inception Score 표준편차
        """
        N = len(images)
        
        # Get predictions
        preds = []
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE].to(self.device)
                batch = self.transform(batch)
                pred = self.inception_model(batch)
                pred = F.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Calculate IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)

class FIDCalculator:
    """Frechet Inception Distance 계산"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # weights=Inception_V3_Weights.IMAGENET1K_V1
        self.inception_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        ).to(device)
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_activations(self, images):
        """
        이미지에서 Inception 특징 추출
        Args:
            images: (N, 3, H, W) tensor, 값 범위 [0, 1]
        Returns:
            activations: (N, 2048) numpy array
        """
        activations = []
        
        with torch.no_grad():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE].to(self.device)
                batch = self.transform(batch)
                
                pred = self.inception_model(batch)
                activations.append(pred.cpu().numpy())
        
        activations = np.concatenate(activations, axis=0)
        return activations
    
    def calculate_fid(self, real_images, fake_images):
        """
        FID 계산
        Args:
            real_images: (N, 3, H, W) tensor
            fake_images: (M, 3, H, W) tensor
        Returns:
            fid_score: FID 값
        """
        # Get activations
        print("  Extracting real image features...")
        act_real = self.get_activations(real_images)
        
        print("  Extracting fake image features...")
        act_fake = self.get_activations(fake_images)
        
        # Calculate statistics
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake, rowvar=False)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid
    
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Frechet distance 계산"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # 수정: disp=False 제거
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 수정: pretrained=True → weights=VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])
        self.slice5 = nn.Sequential(*list(vgg.children())[21:30])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.criterion = nn.L1Loss()
    
    def forward(self, x, y):
        x_vgg = self.normalize(x)
        y_vgg = self.normalize(y)
        
        loss = 0
        for i in range(1, 6):
            slice_fn = getattr(self, f'slice{i}')
            x_vgg = slice_fn(x_vgg)
            y_vgg = slice_fn(y_vgg)
            loss += self.weights[i-1] * self.criterion(x_vgg, y_vgg)
        
        return loss
    
    def normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x * 0.5 + 0.5 - mean) / std

class LPIPSCalculator:
    """Learned Perceptual Image Patch Similarity 계산"""
    def __init__(self, device='cuda', net='alex'):
        """
        Args:
            device: 'cuda' or 'cpu'
            net: 'alex', 'vgg', 'squeeze' 중 선택
        """
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
    
    def calculate_lpips(self, real_images, fake_images):
        """
        LPIPS 계산
        Args:
            real_images: (N, 3, H, W) tensor, 값 범위 [-1, 1] or [0, 1]
            fake_images: (N, 3, H, W) tensor, 값 범위 [-1, 1] or [0, 1]
        Returns:
            lpips_score: LPIPS 평균값
        """
        lpips_scores = []
        
        with torch.no_grad():
            for i in range(len(real_images)):
                real = real_images[i:i+1].to(self.device)
                fake = fake_images[i:i+1].to(self.device)
                
                # LPIPS는 [-1, 1] 범위 기대
                if real.max() <= 1.0 and real.min() >= 0.0:
                    real = real * 2 - 1
                if fake.max() <= 1.0 and fake.min() >= 0.0:
                    fake = fake * 2 - 1
                
                score = self.loss_fn(real, fake)
                lpips_scores.append(score.item())
        
        return np.mean(lpips_scores), np.std(lpips_scores)

# ============================================
# 평가 함수
# ============================================
def evaluate_tooth_generation(netG, val_loader, tooth_type, tooth_position, epoch):
    """
    특정 치아 위치에 대한 생성 품질 평가
    Args:
        netG: Generator 모델
        val_loader: Validation DataLoader
        tooth_type: 'lower' or 'upper'
        tooth_position: 치아 위치 번호
        epoch: 현재 epoch
    Returns:
        metrics: dict with FID, IS, LPIPS scores
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Tooth #{tooth_position} - Epoch {epoch+1}")
    print(f"{'='*60}")
    
    netG.eval()
    
    # 메트릭 계산기 초기화
    fid_calc = FIDCalculator(device=device)
    is_calc = InceptionScoreCalculator(device=device)
    lpips_calc = LPIPSCalculator(device=device)
    
    # 이미지 수집
    real_images = []
    fake_images = []
    normal_images = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting images"):
            normal_img = batch['normal_img'].to(device)
            normal_mask = batch['normal_mask'].to(device)
            decayed_img = batch['decayed_img'].to(device)
            
            # Generate fake images
            fake_decayed = netG(normal_img, normal_mask)
            
            # Denormalize to [0, 1] for metrics
            normal_denorm = (normal_img + 1) / 2.0
            fake_denorm = torch.clamp((fake_decayed + 1) / 2.0, 0, 1)
            real_denorm = (decayed_img + 1) / 2.0
            
            normal_images.append(normal_denorm.cpu())
            fake_images.append(fake_denorm.cpu())
            real_images.append(real_denorm.cpu())
    
    # Concatenate all images
    normal_images = torch.cat(normal_images, dim=0)
    fake_images = torch.cat(fake_images, dim=0)
    real_images = torch.cat(real_images, dim=0)
    
    print(f"\nCollected {len(fake_images)} images")
    
    # Calculate FID
    print("\n[1/3] Calculating FID...")
    try:
        fid_score = fid_calc.calculate_fid(real_images, fake_images)
        print(f"  FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"  FID calculation failed: {e}")
        fid_score = None
    
    # Calculate Inception Score
    print("\n[2/3] Calculating Inception Score...")
    try:
        is_mean, is_std = is_calc.calculate_inception_score(fake_images)
        print(f"  IS Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"  IS calculation failed: {e}")
        is_mean, is_std = None, None
    
    # Calculate LPIPS
    print("\n[3/3] Calculating LPIPS...")
    try:
        # LPIPS는 paired comparison이므로 normal vs fake 비교
        lpips_mean, lpips_std = lpips_calc.calculate_lpips(normal_images, fake_images)
        print(f"  LPIPS Score: {lpips_mean:.4f} ± {lpips_std:.4f}")
    except Exception as e:
        print(f"  LPIPS calculation failed: {e}")
        lpips_mean, lpips_std = None, None
    
    # 결과 정리
    metrics = {
        'tooth_type': tooth_type,
        'tooth_position': tooth_position,
        'epoch': epoch + 1,
        'fid': float(fid_score) if fid_score is not None else None, # float() 추가
        'is_mean': float(is_mean) if is_mean is not None else None, # float() 추가
        'is_std': float(is_std) if is_std is not None else None,   # float() 추가
        'lpips_mean': float(lpips_mean) if lpips_mean is not None else None, # float() 추가
        'lpips_std': float(lpips_std) if lpips_std is not None else None,   # float() 추가
        'num_images': len(fake_images)
    }
    
    # 결과 저장
    save_metrics(metrics, tooth_type, tooth_position, epoch)
    
    return metrics

def save_metrics(metrics, tooth_type, tooth_position, epoch):
    """메트릭 결과 저장"""
    import json
    import csv
    
    # JSON 저장
    json_path = f"{METRICS_DIR}/{tooth_type}_tooth{tooth_position}_epoch{epoch+1}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # CSV에 append (전체 메트릭 히스토리)
    csv_path = f"{METRICS_DIR}/{tooth_type}_tooth{tooth_position}_metrics_history.csv"
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    print(f"\nMetrics saved to {json_path}")

def plot_metrics_comparison(tooth_type='lower'):
    """치아 위치별 메트릭 비교 그래프 생성"""
    import pandas as pd
    
    # 모든 치아의 최종 메트릭 수집
    if tooth_type == 'lower':
        positions = [31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]
    else:
        positions = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]
    
    all_metrics = []
    
    for pos in positions:
        csv_path = f"{METRICS_DIR}/{tooth_type}_tooth{pos}_metrics_history.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                # 마지막 epoch 메트릭 사용
                last_metrics = df.iloc[-1].to_dict()
                all_metrics.append(last_metrics)
    
    if not all_metrics:
        print("No metrics found for comparison")
        return
    
    df_all = pd.DataFrame(all_metrics)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # FID Score
    if 'fid' in df_all.columns:
        axes[0, 0].bar(df_all['tooth_position'], df_all['fid'], color='steelblue')
        axes[0, 0].set_xlabel('Tooth Position', fontsize=12)
        axes[0, 0].set_ylabel('FID Score (Lower is Better)', fontsize=12)
        axes[0, 0].set_title('FID Score by Tooth Position', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Inception Score
    if 'is_mean' in df_all.columns:
        axes[0, 1].bar(df_all['tooth_position'], df_all['is_mean'], 
                       yerr=df_all.get('is_std', 0), color='coral', capsize=5)
        axes[0, 1].set_xlabel('Tooth Position', fontsize=12)
        axes[0, 1].set_ylabel('IS Score (Higher is Better)', fontsize=12)
        axes[0, 1].set_title('Inception Score by Tooth Position', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # LPIPS Score
    if 'lpips_mean' in df_all.columns:
        axes[1, 0].bar(df_all['tooth_position'], df_all['lpips_mean'], 
                       yerr=df_all.get('lpips_std', 0), color='mediumseagreen', capsize=5)
        axes[1, 0].set_xlabel('Tooth Position', fontsize=12)
        axes[1, 0].set_ylabel('LPIPS Score (Lower is Better)', fontsize=12)
        axes[1, 0].set_title('LPIPS Score by Tooth Position', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Summary table
    axes[1, 1].axis('off')
    summary_text = "Summary Statistics\n\n"
    
    if 'fid' in df_all.columns:
        summary_text += f"FID - Mean: {df_all['fid'].mean():.2f}, Std: {df_all['fid'].std():.2f}\n"
    if 'is_mean' in df_all.columns:
        summary_text += f"IS - Mean: {df_all['is_mean'].mean():.2f}, Std: {df_all['is_mean'].std():.2f}\n"
    if 'lpips_mean' in df_all.columns:
        summary_text += f"LPIPS - Mean: {df_all['lpips_mean'].mean():.4f}, Std: {df_all['lpips_mean'].std():.4f}\n"
    
    summary_text += f"\nTotal Teeth Evaluated: {len(df_all)}"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{tooth_type.upper()} Teeth - Generation Quality Metrics', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f"{METRICS_DIR}/{tooth_type}_teeth_metrics_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to {save_path}")

def generate_metrics_report():
    """전체 메트릭 리포트 생성"""
    import pandas as pd
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DENTAL CARIES GENERATION - METRICS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    for tooth_type in ['lower', 'upper']:
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"{tooth_type.upper()} TEETH")
        report_lines.append(f"{'='*80}")
        
        if tooth_type == 'lower':
            positions = [31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]
        else:
            positions = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]
        
        all_metrics = []
        for pos in positions:
            csv_path = f"{METRICS_DIR}/{tooth_type}_tooth{pos}_metrics_history.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    last_metrics = df.iloc[-1].to_dict()
                    all_metrics.append(last_metrics)
        
        if all_metrics:
            df_all = pd.DataFrame(all_metrics)
            
            report_lines.append(f"\nTooth Position | FID Score | IS Score | LPIPS Score")
            report_lines.append("-" * 60)
            
            for _, row in df_all.iterrows():
                pos = int(row['tooth_position'])
                fid = row.get('fid', 'N/A')
                is_score = row.get('is_mean', 'N/A')
                lpips = row.get('lpips_mean', 'N/A')
                
                fid_str = f"{fid:.2f}" if isinstance(fid, (int, float)) else str(fid)
                is_str = f"{is_score:.2f}" if isinstance(is_score, (int, float)) else str(is_score)
                lpips_str = f"{lpips:.4f}" if isinstance(lpips, (int, float)) else str(lpips)
                
                report_lines.append(f"    #{pos:2d}      |  {fid_str:>7s}  |  {is_str:>6s}  |   {lpips_str:>7s}")
            
            report_lines.append("-" * 60)
            
            # 평균값
            if 'fid' in df_all.columns:
                fid_mean = df_all['fid'].mean()
                report_lines.append(f"\nAverage FID: {fid_mean:.2f}")
            if 'is_mean' in df_all.columns:
                is_mean = df_all['is_mean'].mean()
                report_lines.append(f"Average IS: {is_mean:.2f}")
            if 'lpips_mean' in df_all.columns:
                lpips_mean = df_all['lpips_mean'].mean()
                report_lines.append(f"Average LPIPS: {lpips_mean:.4f}")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("REPORT GENERATED")
    report_lines.append("="*80)
    
    # 리포트 저장
    report_path = f"{METRICS_DIR}/metrics_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # 콘솔 출력
    print('\n'.join(report_lines))
    print(f"\nReport saved to {report_path}")

# ============================================
# Dataset (기존과 동일)
# ============================================
class ToothPositionDataset(Dataset):
    """STRICT MODE: 같은 치아 위치만 사용"""
    def __init__(self, metadata_file, data_dir, tooth_type='lower', 
                 tooth_position=31, transform=None, mode='train', debug=False):
        self.data_dir = data_dir
        self.tooth_type = tooth_type
        self.tooth_position = tooth_position
        self.transform = transform
        self.mode = mode
        self.debug = debug
        
        if tooth_type == 'lower':
            self.image_dir = IMAGE_DIR_LOWER
            label_dir = LABEL_DIR_LOWER
            # Lower: 12개 치아 (31-36, 41-46) # 20251128 추가
            self.required_positions = [31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]            
        else:
            self.image_dir = IMAGE_DIR_UPPER
            label_dir = LABEL_DIR_UPPER
            # Upper: 12개 치아 (11-16, 21-26) # 20251128 추가
            self.required_positions = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]            
        
        print(f"\n{'='*60}")
        print(f"STRICT Dataset - {tooth_type.upper()} #{tooth_position} ({mode})")
        print(f"Only using tooth #{tooth_position} decayed data")
        print(f"{'='*60}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        all_valid_images = []
        for item in metadata['info']:
            filepath = item['image_filepath']
            if tooth_type in filepath.lower():
                filename = Path(filepath).stem
                all_valid_images.append(filename)
        
        np.random.seed(42)
        shuffled = np.array(all_valid_images)
        np.random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * 0.8)
        if mode == 'train':
            self.valid_images = list(shuffled[:split_idx])
        else:
            self.valid_images = list(shuffled[split_idx:])
        
        # 12개 치아 완전한 이미지만 필터링 # 20251128 추가
        self.complete_images = self._filter_complete_images(label_dir)
        print(f"  Images with ALL 12 teeth: {len(self.complete_images)}")
        print(f"  Filtered out: {len(self.valid_images) - len(self.complete_images)}")
        
        # 완전한 이미지만 사용 # 20251128 추가
        self.valid_images = self.complete_images
        
        # 같은 위치만 추출
        self.normal_teeth, self.decayed_teeth = \
            self._extract_same_position_only(label_dir, tooth_position)
        
        print(f"  Normal teeth (#{tooth_position}): {len(self.normal_teeth)}")
        print(f"  Decayed teeth (#{tooth_position}): {len(self.decayed_teeth)}")
        
        # 경고 메시지
        if len(self.normal_teeth) == 0:
            print(f"  ❌ No normal teeth found!")
        if len(self.decayed_teeth) == 0:
            print(f"  ❌ No decayed teeth found!")
            print(f"  ⚠️  This position CANNOT be trained in STRICT mode!")

    def _filter_complete_images(self, label_dir): # 20251128 추가
        """12개 치아가 모두 있는 이미지만 필터링"""
        complete_images = []
        
        json_files = sorted(Path(label_dir).glob('*.json'))
        
        for json_file in tqdm(json_files, desc=f"Filtering complete images"):
            file_stem = json_file.stem
            if file_stem not in self.valid_images:
                continue
            
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            
            # 이미지에 있는 치아 위치 수집
            tooth_positions_in_image = set()
            for tooth_info in label_data['tooth']:
                teeth_num = tooth_info['teeth_num']
                tooth_positions_in_image.add(teeth_num)
            
            # 12개 치아가 모두 있는지 확인
            required_set = set(self.required_positions)
            if required_set.issubset(tooth_positions_in_image):
                complete_images.append(file_stem)
        
        return complete_images
    
    def _extract_same_position_only(self, label_dir, tooth_position):
        """같은 위치만 추출"""
        normal_teeth = []
        decayed_teeth = []
        
        json_files = sorted(Path(label_dir).glob('*.json'))
        
        for json_file in tqdm(json_files, desc=f"Loading #{tooth_position}"):
            file_stem = json_file.stem
            if file_stem not in self.valid_images:
                continue
            
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            
            full_image_path = os.path.join(self.image_dir, f"{file_stem}.png")
            if not os.path.exists(full_image_path):
                continue
            
            full_image = cv2.imread(full_image_path)
            if full_image is None:
                continue
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            
            for tooth_info in label_data['tooth']:
                teeth_num = tooth_info['teeth_num']
                
                # 핵심: 같은 위치만 필터링
                if teeth_num != tooth_position:
                    continue
                
                segmentation = tooth_info['segmentation']
                decayed = tooth_info['decayed']
                
                mask = self._seg_to_mask(segmentation, full_image.shape[:2])
                bbox = self._get_bbox(mask)
                if bbox is None:
                    continue
                
                tooth_img, tooth_mask = self._crop_with_mask(full_image, bbox, mask)
                if tooth_img is None:
                    continue
                
                tooth_data = {
                    'image': tooth_img,
                    'mask': tooth_mask,
                    'teeth_num': teeth_num,
                    'decayed': decayed,
                    'source_file': file_stem
                }
                
                if decayed:
                    decayed_teeth.append(tooth_data)
                else:
                    normal_teeth.append(tooth_data)
        
        return normal_teeth, decayed_teeth
    
    def _seg_to_mask(self, segmentation, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        if isinstance(segmentation[0], list):
            for seg in segmentation:
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 255)
        else:
            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask
    
    def _get_bbox(self, mask):
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        padding = int(max(w, h) * 0.1)
        return (max(0, x-padding), max(0, y-padding), w+2*padding, h+2*padding)
    
    def _crop_with_mask(self, image, bbox, mask):
        x, y, w, h = bbox
        
        if x + w > image.shape[1]:
            w = image.shape[1] - x
        if y + h > image.shape[0]:
            h = image.shape[0] - y
        
        if w <= 0 or h <= 0:
            return None, None
        
        cropped_img = image[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w].copy()
        
        if cropped_img.shape[0] < 20 or cropped_img.shape[1] < 20:
            return None, None
        
        return cropped_img, cropped_mask
    
    def __len__(self):
        # 정상과 충치 모두 있어야 학습 가능
        if len(self.normal_teeth) > 0 and len(self.decayed_teeth) > 0:
            return min(len(self.normal_teeth) * 10, 1000)
        else:
            return 0
    
    def __getitem__(self, idx):
        # 검증: 정상과 충치 모두 필요
        if len(self.normal_teeth) == 0 or len(self.decayed_teeth) == 0:
            raise ValueError(
                f"Cannot train tooth #{self.tooth_position}: "
                f"Normal={len(self.normal_teeth)}, Decayed={len(self.decayed_teeth)}"
            )
        
        # 같은 위치의 정상치아
        normal_idx = np.random.randint(len(self.normal_teeth))
        normal_tooth = self.normal_teeth[normal_idx]
        
        # 같은 위치의 충치치아만 사용 (다른 위치 사용 안함!)
        decayed_idx = np.random.randint(len(self.decayed_teeth))
        decayed_tooth = self.decayed_teeth[decayed_idx]
        
        normal_img = Image.fromarray(normal_tooth['image'])
        normal_mask = Image.fromarray(normal_tooth['mask'])
        decayed_img = Image.fromarray(decayed_tooth['image'])
        decayed_mask = Image.fromarray(decayed_tooth['mask'])
        
        if self.transform:
            normal_img = self.transform(normal_img)
            decayed_img = self.transform(decayed_img)
            
            mask_transform = transforms.Compose([
                transforms.Resize((TOOTH_SIZE, TOOTH_SIZE)),
                transforms.ToTensor(),
            ])
            normal_mask = mask_transform(normal_mask)
            decayed_mask = mask_transform(decayed_mask)
        
        return {
            'normal_img': normal_img,
            'normal_mask': normal_mask,
            'decayed_img': decayed_img,
            'decayed_mask': decayed_mask,
            'teeth_num': self.tooth_position,
            'source_file': normal_tooth.get('source_file', 'unknown')
        }

# 20251128 추가
class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual Loss"""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])# relu4_4
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
    
    def forward(self, x, y):
        # 정규화: [-1, 1] → [0, 1] → VGG normalization
        x = (x + 1) / 2.0
        y = (y + 1) / 2.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0
        
        x = self.slice1(x)
        y = self.slice1(y)
        loss += self.criterion(x, y)
        
        x = self.slice2(x)
        y = self.slice2(y)
        loss += self.criterion(x, y)
        
        x = self.slice3(x)
        y = self.slice3(y)
        loss += self.criterion(x, y)
        
        x = self.slice4(x)
        y = self.slice4(y)
        loss += self.criterion(x, y)
        
        return loss

# ============================================
# InstaGAN Models (기존과 동일)
# ============================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class InstaGANGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_blocks=9): # 20251128 추가 (n_block=6 -> n_block=9)
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResidualBlock(ngf * 4))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 수정: ConvTranspose2d → Upsample + Conv : # 20251128 추가
        self.decoder = nn.Sequential(
            # 수정:
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 2, 3, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf, 3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        )
    
    def forward(self, x, mask):
        input_with_mask = torch.cat([x, mask], dim=1)
        feat = self.encoder(input_with_mask)
        feat = self.res_blocks(feat)
        output = self.decoder(feat)
        return output

class InstaGANDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x, mask):
        input_with_mask = torch.cat([x, mask], dim=1)
        return self.model(input_with_mask)

# ============================================
# Training Functions (메트릭 평가 추가)
# ============================================
def train_tooth_position(tooth_type='lower', tooth_position=31, num_epochs=NUM_EPOCHS):
    """STRICT MODE 학습"""
    print(f"\n{'='*60}")
    print(f"STRICT InstaGAN Training - {tooth_type.upper()} Tooth #{tooth_position}")
    print(f"Only using tooth #{tooth_position} decayed data")
    print(f"{'='*60}\n")
    
    transform = transforms.Compose([
        transforms.Resize((TOOTH_SIZE, TOOTH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ToothPositionDataset(
        METADATA_FILE, DATA_DIR,
        tooth_type=tooth_type,
        tooth_position=tooth_position,
        transform=transform,
        mode='train',
        debug=True
    )
    
    # 검증: 최소 5개 필요
    if len(train_dataset) < 5:
        print(f"\n{'='*60}")
        print(f"❌ SKIPPING Tooth #{tooth_position}")
        print(f"Reason: Insufficient data (need both normal AND decayed)")
        print(f"Available: {len(train_dataset)} pairs")
        print(f"{'='*60}\n")
        return False
    
    val_dataset = ToothPositionDataset(
        METADATA_FILE, DATA_DIR,
        tooth_type=tooth_type,
        tooth_position=tooth_position,
        transform=transform,
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), 
                             shuffle=True, num_workers=4, drop_last=True)
    
    val_loader = None
    if len(val_dataset) >= 5:
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    netG = InstaGANGenerator().to(device)
    netD = InstaGANDiscriminator().to(device)
    
    criterionGAN = nn.MSELoss()
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGPerceptualLoss().to(device) # 20251128 추가 (VGG Loss)
    
    # 20251128 추가 Optimizer 수정
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        # ============================================
        # Discriminator 업데이트 빈도 조정 # 20251128 추가
        # ============================================
        N_CRITIC = 1  # Discriminator를 2번 업데이트

        for batch in pbar:
            normal_img = batch['normal_img'].to(device)
            normal_mask = batch['normal_mask'].to(device)
            decayed_img = batch['decayed_img'].to(device)
            decayed_mask = batch['decayed_mask'].to(device)
            
            # Discriminator 업데이트 # 20251128 추가
            for _ in range(N_CRITIC):
                optimizer_D.zero_grad()
                pred_real = netD(decayed_img, decayed_mask)
                loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
                
                fake_decayed = netG(normal_img, normal_mask)
                pred_fake = netD(fake_decayed.detach(), normal_mask)
                loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
            
            # Generator 1번 업데이트
            optimizer_G.zero_grad()
            fake_decayed = netG(normal_img, normal_mask)
            pred_fake = netD(fake_decayed, normal_mask)

            # 수정 : VGG loss 추가, 가중치 조정
            loss_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_L1 = criterionL1(fake_decayed, decayed_img)
            loss_instance = criterionL1(fake_decayed * normal_mask, decayed_img * normal_mask)
            loss_perceptual = criterionVGG(fake_decayed, decayed_img)  # 20251128 추가

            # lambda 값 적용  # 20251128 추가
            loss_G = loss_GAN + 10.0 * loss_L1 + 5.0 * loss_instance + 1.0 * loss_perceptual
            loss_G.backward()
            optimizer_G.step()            
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            # 20251128 추가 : VGG loss도 출력
            pbar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D': f'{loss_D.item():.4f}',
                'L1': f'{loss_L1.item():.4f}',
                'VGG': f'{loss_perceptual.item():.4f}'
            })
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        
        print(f"Epoch {epoch+1} - G: {avg_loss_G:.4f}, D: {avg_loss_D:.4f}")
        
        if (epoch + 1) % 50 == 0:
            if val_loader is not None:
                validate_and_save(netG, val_loader, epoch, tooth_type, tooth_position)
                
                if len(val_dataset) >= 10:
                    try:
                        evaluate_tooth_generation(netG, val_loader, tooth_type, tooth_position, epoch)
                    except Exception as e:
                        print(f"⚠️  Metrics evaluation failed: {e}")
            
            torch.save({
                'epoch': epoch,
                'tooth_position': tooth_position,
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, f'{CHECKPOINT_DIR}/strict_{tooth_type}_tooth{tooth_position}_epoch_{epoch+1}.pth')
    
    return True

def validate_and_save(netG, val_loader, epoch, tooth_type, tooth_position):
    """검증 및 결과 저장"""
    netG.eval()
    save_dir = f'{RESULTS_DIR}/{tooth_type}_tooth{tooth_position}_epoch_{epoch+1}'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:
                break
            
            normal_img = batch['normal_img'].to(device)
            normal_mask = batch['normal_mask'].to(device)
            decayed_img = batch['decayed_img'].to(device)
            source_files = batch['source_file']
            
            fake_decayed = netG(normal_img, normal_mask)
            
            normal_display = torch.clamp((normal_img + 1) / 2.0, 0, 1)
            fake_display = torch.clamp((fake_decayed + 1) / 2.0, 0, 1)
            decayed_display = torch.clamp((decayed_img + 1) / 2.0, 0, 1)
            mask_display = normal_mask.repeat(1, 3, 1, 1)
            
            for j in range(min(4, normal_img.size(0))):
                source_file = source_files[j] if isinstance(source_files, (list, tuple)) else source_files
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                axes[0, 0].imshow(normal_display[j].cpu().permute(1, 2, 0).numpy())
                axes[0, 0].set_title(f'Normal Tooth #{tooth_position}', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(fake_display[j].cpu().permute(1, 2, 0).numpy())
                axes[0, 1].set_title(f'Generated Decayed', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(decayed_display[j].cpu().permute(1, 2, 0).numpy())
                axes[1, 0].set_title(f'Real Decayed #{tooth_position}', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(mask_display[j].cpu().permute(1, 2, 0).numpy(), cmap='gray')
                axes[1, 1].set_title(f'Mask', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                plt.suptitle(f'Tooth #{tooth_position} | Source: {source_file} | Epoch: {epoch+1}', 
                           fontsize=10, y=0.98)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/tooth{tooth_position}_batch{i}_sample{j}.png', 
                          dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"Saved to {save_dir}")

# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("STRICT MODE - InstaGAN Dental Caries Generation")
    print("Only using same position decayed teeth")
    print("="*60)
    
    lower_positions = [31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]
    upper_positions = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]
    
    trained_lower = []
    skipped_lower = []
    
    print("\n[Phase 1] Training Lower Teeth (STRICT)")
    for tooth_pos in lower_positions:
        try:
            success = train_tooth_position(tooth_type='lower', tooth_position=tooth_pos, num_epochs=NUM_EPOCHS)
            if success:
                trained_lower.append(tooth_pos)
            else:
                skipped_lower.append(tooth_pos)
        except Exception as e:
            print(f"❌ Error training tooth #{tooth_pos}: {e}")
            import traceback
            traceback.print_exc()
            skipped_lower.append(tooth_pos)
            continue
    
    print(f"\n{'='*60}")
    print(f"Lower Teeth Summary")
    print(f"{'='*60}")
    print(f"Successfully trained: {trained_lower}")
    print(f"⚠️  Skipped (no decayed data): {skipped_lower}")
    
    if trained_lower:
        try:
            plot_metrics_comparison(tooth_type='lower')
        except Exception as e:
            print(f"⚠️  Failed: {e}")
    
    trained_upper = []
    skipped_upper = []
    
    print("\n[Phase 2] Training Upper Teeth (STRICT)")
    for tooth_pos in upper_positions:
        try:
            success = train_tooth_position(tooth_type='upper', tooth_position=tooth_pos, num_epochs=NUM_EPOCHS)
            if success:
                trained_upper.append(tooth_pos)
            else:
                skipped_upper.append(tooth_pos)
        except Exception as e:
            print(f"❌ Error training tooth #{tooth_pos}: {e}")
            import traceback
            traceback.print_exc()
            skipped_upper.append(tooth_pos)
            continue
    
    print(f"\n{'='*60}")
    print(f"Upper Teeth Summary")
    print(f"{'='*60}")
    print(f"Successfully trained: {trained_upper}")
    print(f"⚠️  Skipped (no decayed data): {skipped_upper}")
    
    if trained_upper:
        try:
            plot_metrics_comparison(tooth_type='upper')
        except Exception as e:
            print(f"⚠️  Failed: {e}")
    
    print("\n[Generating Final Report]")
    try:
        generate_metrics_report()
    except Exception as e:
        print(f"⚠️  Failed: {e}")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY - STRICT MODE")
    print("="*60)
    print(f"Total trained: {len(trained_lower) + len(trained_upper)}")
    print(f"  Lower: {len(trained_lower)}/{len(lower_positions)}")
    print(f"  Upper: {len(trained_upper)}/{len(upper_positions)}")
    print(f"\nTotal skipped: {len(skipped_lower) + len(skipped_upper)}")
    print(f"  Lower: {len(skipped_lower)}/{len(lower_positions)}")
    print(f"  Upper: {len(skipped_upper)}/{len(upper_positions)}")
    print("\nSTRICT MODE training completed!")

## 📊 **추가된 기능 요약**

# ### 1. **FID (Frechet Inception Distance)**
# - 실제 충치 이미지 분포와 생성된 충치 이미지 분포 간의 거리 측정
# - **낮을수록 좋음** (실제와 유사)

# ### 2. **IS (Inception Score)**
# - 생성된 이미지의 품질과 다양성 평가
# - **높을수록 좋음** (명확하고 다양한 생성)

# ### 3. **LPIPS (Learned Perceptual Image Patch Similarity)**
# - 인간의 지각적 유사도 측정
# - **낮을수록 좋음** (정상치아와 생성치아의 차이가 적절)

# ### 4. **자동 평가 및 리포트 생성**
# - 50 epoch마다 자동 메트릭 계산
# - 치아 위치별 비교 그래프 생성
# - 전체 메트릭 리포트 텍스트 파일 생성

# ### 5. **저장 구조**
# ```
# instagan_teeth_by_position/
# ├── metrics/
# │   ├── lower_tooth31_epoch50_metrics.json
# │   ├── lower_tooth31_metrics_history.csv
# │   ├── lower_teeth_metrics_comparison.png
# │   ├── upper_teeth_metrics_comparison.png
# │   └── metrics_report.txt

# ============================================
# MASK 설명
# ============================================

# # Line 1004: mask_display 생성
# mask_display = normal_mask.repeat(1, 3, 1, 1)

# # Line 1023: 시각화
# axes[1, 1].imshow(mask_display[j].cpu().permute(1, 2, 0).numpy(), cmap='gray')
# axes[1, 1].set_title(f'Mask', fontsize=12, fontweight='bold')

# # """
# 이 MASK는 무엇인가?
# ====================

# 1. **정의**: Segmentation Mask (치아 영역 마스크)
#    - 치아 영역: 흰색 (255 또는 1.0)
#    - 배경 영역: 검은색 (0)

# 2. **생성 과정**:
#    # Line 640-660: _seg_to_mask 메서드
#    def _seg_to_mask(self, segmentation, image_shape):
#        mask = np.zeros(image_shape, dtype=np.uint8)  # 검은 배경
#        if isinstance(segmentation[0], list):
#            for seg in segmentation:
#                points = np.array(seg).reshape(-1, 2).astype(np.int32)
#                cv2.fillPoly(mask, [points], 255)  # 치아 영역을 흰색으로
#        else:
#            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
#            cv2.fillPoly(mask, [points], 255)
#        return mask

# 3. **용도**:
#    - Generator 입력: `torch.cat([x, mask], dim=1)` (RGB 3채널 + Mask 1채널)
#    - Instance-aware Loss: `loss_instance = criterionL1(fake * mask, real * mask)`
#    - 치아 영역만 집중해서 학습

# 4. **이미지에서 보이는 것**:
#    - 흰색 영역: 치아 #13의 경계 (segmentation polygon)
#    - 검은색 영역: 배경 (치아가 아닌 부분)
#    - 이 마스크를 사용해 치아 영역만 정확하게 학습

# 5. **왜 필요한가?**:
#    - 배경(잇몸, 입술 등)은 학습하지 않고 치아만 학습
#    - Instance-aware: 각 치아를 개별 객체로 인식
#    - InstaGAN의 핵심 아이디어!
# """

# # ============================================
# # 최종 권장 하이퍼파라미터
# # ============================================

# # 이미지 품질 개선
# TOOTH_SIZE = 128              # 해상도 증가
# BATCH_SIZE = 8                # 메모리 고려
# NUM_EPOCHS = 500              # 충분한 학습
# LR = 0.0001                   # 안정적 학습

# # 모델 구조
# ngf = 64                      # Generator 필터
# n_blocks = 9                  # ResBlock 증가
# ndf = 64                      # Discriminator 필터

# # Loss 가중치
# lambda_GAN = 1.0              # GAN loss
# lambda_L1 = 20.0              # Pixel-wise (증가)
# lambda_instance = 10.0        # Instance-aware (증가)

# # 학습 전략
# N_CRITIC = 1                  # D를 1번 업데이트
# use_augmentation = True       # Data augmentation