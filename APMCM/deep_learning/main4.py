# 导入所需的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# 数据集定义
class UnderwaterDataset(Dataset):
    """
    自定义数据集类，用于加载退化图像及其对应的清晰图像。
    """
    def __init__(self, degraded_dir, clear_dir, transform=None):
        if not os.path.exists(degraded_dir) or not os.path.exists(clear_dir):
            raise ValueError("目录不存在，请检查路径")
            
        self.degraded_dir = degraded_dir
        self.clear_dir = clear_dir
        self.degraded_images = sorted(os.listdir(degraded_dir))
        self.clear_images = sorted(os.listdir(clear_dir))
        
        if len(self.degraded_images) != len(self.clear_images):
            raise ValueError("退化图像和清晰图像数量不匹配")
            
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        try:
            # 读取图像
            degraded_path = os.path.join(self.degraded_dir, self.degraded_images[idx])
            clear_path = os.path.join(self.clear_dir, self.clear_images[idx])
            
            if not os.path.exists(degraded_path) or not os.path.exists(clear_path):
                raise FileNotFoundError(f"图像文件不存在: {degraded_path} 或 {clear_path}")

            # 打开并转换图像
            degraded = Image.open(degraded_path).convert('RGB')
            clear = Image.open(clear_path).convert('RGB')
            
            # 检查图像是否有效
            if degraded.size == (0, 0) or clear.size == (0, 0):
                raise ValueError(f"无效的图像大小: {self.degraded_images[idx]} 或 {self.clear_images[idx]}")

            # 应用变换
            if self.transform:
                degraded = self.transform(degraded)
                clear = self.transform(clear)
                
                # 确保输出是张量
                if not isinstance(degraded, torch.Tensor) or not isinstance(clear, torch.Tensor):
                    raise TypeError("转换后的数据必须是torch.Tensor类型")

            return degraded, clear
            
        except Exception as e:
            print(f"加载图像 {idx} 时出错: {str(e)}")
            # 返回一个有效的替代值，而不是None
            return torch.zeros((3, 256, 256)), torch.zeros((3, 256, 256))

# 数据加载
def load_data(degraded_dir, clear_dir, batch_size=8):
    """
    数据加载函数，返回 DataLoader 对象。
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),         # 转换为张量
    ])
    dataset = UnderwaterDataset(degraded_dir, clear_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 数据集划分函数
def create_train_test_dataloaders(degraded_dir, clear_dir, batch_size=8, test_split=0.2):
    """
    创建训练集和测试集的数据加载器
    
    参数:
        degraded_dir: 退化图像目录
        clear_dir: 清晰图像目录
        batch_size: 批次大小
        test_split: 测试集占比
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建完整数据集
    dataset = UnderwaterDataset(
        degraded_dir=degraded_dir,
        clear_dir=clear_dir,
        transform=transform
    )
    
    # 计算划分大小
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    # 随机划分数据集
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子确保可复现
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

def create_dataloader(degraded_dir, clear_dir, batch_size=8, shuffle=True):
    """创建数据加载器并验证数据集"""
    # 检查目录是否存在
    if not os.path.exists(degraded_dir):
        raise ValueError(f"退化图像目录不存在: {degraded_dir}")
    if not os.path.exists(clear_dir):
        raise ValueError(f"清晰图像目录不存在: {clear_dir}")
    
    # 获取并检查文件列表
    degraded_files = sorted(os.listdir(degraded_dir))
    clear_files = sorted(os.listdir(clear_dir))
    
    # 打印文件数量信息
    print(f"\n数据集信息:")
    print(f"退化图像目录: {degraded_dir}")
    print(f"清晰图像目录: {clear_dir}")
    print(f"退化图像数量: {len(degraded_files)}")
    print(f"清晰图像数量: {len(clear_files)}")
    
    # 如果数量不匹配，打印前几个文件名以供对比
    if len(degraded_files) != len(clear_files):
        print("\n文件名对比(前5个):")
        print("退化图像:", degraded_files[:5])
        print("清晰图像:", clear_files[:5])
        raise ValueError(f"图像数量不匹配: 退化图像 {len(degraded_files)} 张, 清晰图像 {len(clear_files)} 张")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    dataset = UnderwaterDataset(
        degraded_dir=degraded_dir,
        clear_dir=clear_dir,
        transform=transform
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader

# U-Net 定义
class UNet(nn.Module):
    """
    定义 U-Net 网络，用于水下图像增强。
    """
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(inplace=True),                      # 激活函数
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                             # 池化层
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 转置卷积层
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),  # 通道缩减
            nn.Sigmoid()                       # 激活函数，生成注意力权重
        )

    def forward(self, x):
        x1 = self.encoder(x)                   # 编码
        x2 = self.attention(x1)                # 注意力模块
        x_out = self.decoder(x1 * x2)          # 加权解码
        return x_out

# 模型训练函数
def train_model(dataloader, epochs=20, lr=1e-4):
    """
    模型训练函数。
    """
    # 实例化模型并移动到 GPU
    model = UNet().cuda()
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for degraded, clear in dataloader:
            degraded, clear = degraded.cuda(), clear.cuda()  # 数据移动到 GPU

            optimizer.zero_grad()            # 梯度清零
            output = model(degraded)         # 前向传播
            loss = criterion(output, clear)  # 计算损失
            loss.backward()                  # 反向传播
            optimizer.step()                 # 更新参数

            total_loss += loss.item()        # 累计损失
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    # 保存模型
    torch.save(model.state_dict(), "unet_underwater.pth")
    print("模型已保存至 unet_underwater.pth")
    return model

# 模型评估函数
def evaluate_model(model, dataloader):
    """评估模型性能并可视化指标"""
    model.eval()
    # 存储指标值的列表
    psnr_values = []
    uciqe_values = []
    uiqm_values = []
    
    with torch.no_grad():
        for batch_idx, (degraded, clear) in enumerate(dataloader):
            degraded, clear = degraded.cuda(), clear.cuda()
            output = model(degraded)
            
            # 处理每个batch中的图像
            for i in range(output.shape[0]):
                # 转换单个图像
                img_output = output[i].cpu().numpy().transpose(1, 2, 0)
                img_clear = clear[i].cpu().numpy().transpose(1, 2, 0)
                img_output = np.clip(img_output * 255.0, 0, 255).astype(np.uint8)
                img_clear = np.clip(img_clear * 255.0, 0, 255).astype(np.uint8)
                
                # 计算指标
                psnr = calculate_psnr(img_clear, img_output)
                uciqe = calculate_uciqe(img_output)
                uiqm = calculate_uiqm(img_output)
                
                # 存储指标值
                psnr_values.append(psnr)
                uciqe_values.append(uciqe)
                uiqm_values.append(uiqm)
                
                print(f"Image {batch_idx*output.shape[0]+i+1}:")
                print(f"PSNR: {psnr:.2f}, UCIQE: {uciqe:.3f}, UIQM: {uiqm:.3f}")

    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_uciqe = np.mean(uciqe_values)
    avg_uiqm = np.mean(uiqm_values)
    
    print("\nAverage Metrics:")
    print(f"PSNR: {avg_psnr:.2f}, UCIQE: {avg_uciqe:.3f}, UIQM: {avg_uiqm:.3f}")

    # 可视化指标
    plt.figure(figsize=(15, 5))
    
    # PSNR图
    plt.subplot(131)
    plt.plot(psnr_values, 'b-')
    plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Average: {avg_psnr:.2f}')
    plt.title('PSNR Values')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR')
    plt.legend()
    
    # UCIQE图
    plt.subplot(132)
    plt.plot(uciqe_values, 'g-')
    plt.axhline(y=avg_uciqe, color='r', linestyle='--', label=f'Average: {avg_uciqe:.3f}')
    plt.title('UCIQE Values')
    plt.xlabel('Image Index')
    plt.ylabel('UCIQE')
    plt.legend()
    
    # UIQM图
    plt.subplot(133)
    plt.plot(uiqm_values, 'm-')
    plt.axhline(y=avg_uiqm, color='r', linestyle='--', label=f'Average: {avg_uiqm:.3f}')
    plt.title('UIQM Values')
    plt.xlabel('Image Index')
    plt.ylabel('UIQM')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_psnr(img1, img2):
    """计算峰值信噪比(PSNR)"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse_r = np.mean((img1[:,:,0] - img2[:,:,0]) ** 2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1]) ** 2)
    mse_b = np.mean((img1[:,:,2] - img2[:,:,2]) ** 2)
    
    mse = (mse_r + mse_g + mse_b) / 3.0
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_uciqe(img):
    """计算水下图像质量评价(UCIQE)"""
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 1. 计算色度
    chroma = np.sqrt(np.square(a) + np.square(b))
    
    # 2. 计算饱和度变异系数
    mu_c = np.mean(chroma)
    std_c = np.std(chroma)
    if (mu_c == 0):
        mu_c = 1e-6
    sat_var = std_c / mu_c
    
    # 3. 计算亮度对比度
    top_15_percent = np.percentile(l, 85)
    bottom_15_percent = np.percentile(l, 15)
    con_l = top_15_percent - bottom_15_percent
    
    # 4. 计算平均饱和度
    avg_sat = np.mean(chroma)
    
    # UCIQE参数
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    uciqe = c1 * sat_var + c2 * con_l + c3 * avg_sat
    return uciqe/100  # 归一化到0-1范围

def calculate_uiqm(img):
    """计算水下图像质量评价指标 UIQM"""
    # 确保图像为 RGB 格式
    if (img.shape[2] == 3):
        img_rgb = img
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    A = A.astype(np.float32) - 128
    B = B.astype(np.float32) - 128

    # 计算 UICM
    RG_var = np.var(A)
    YB_var = np.var(B)
    RG_mean = np.mean(A)
    YB_mean = np.mean(B)
    UICM = -0.0268 * np.sqrt(RG_var + YB_var) + 0.1586 * np.sqrt(RG_mean**2 + YB_mean**2)

    # 计算 UISM
    L = L.astype(np.float32)
    dx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx**2 + dy**2) + 1e-6  # 防止除零
    block_size = 8
    h, w = L.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    EME = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = gradient_magnitude[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            G_max = np.max(block)
            G_min = np.min(block) + 1e-6  # 防止除零
            EME += 20 * np.log10(G_max / G_min)
    UISM = EME / (num_blocks_h * num_blocks_w)

    # 计算 UIConM
    kernel_size = (17, 17)
    mean_local = cv2.blur(L, kernel_size)
    mean_sq_local = cv2.blur(L**2, kernel_size)
    contrast_local = np.sqrt(mean_sq_local - mean_local**2)
    UIConM = np.mean(contrast_local)

    # 计算 UIQM
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM

    return UIQM/100


# 测试函数
def test_model(model, test_loader):
    """在测试集上评估模型"""
    model.eval()
    total_psnr = 0
    total_uciqe = 0
    total_uiqm = 0
    count = 0
    
    with torch.no_grad():
        for degraded, clear in test_loader:
            degraded, clear = degraded.cuda(), clear.cuda()
            output = model(degraded)
            
            for i in range(output.shape[0]):
                img_output = output[i].cpu().numpy().transpose(1, 2, 0)
                img_clear = clear[i].cpu().numpy().transpose(1, 2, 0)
                
                img_output = np.clip(img_output * 255.0, 0, 255).astype(np.uint8)
                img_clear = np.clip(img_clear * 255.0, 0, 255).astype(np.uint8)
                
                # 计算评估指标
                psnr = calculate_psnr(img_clear, img_output)
                uciqe = calculate_uciqe(img_output)
                uiqm = calculate_uiqm(img_output)
                
                total_psnr += psnr
                total_uciqe += uciqe
                total_uiqm += uiqm
                count += 1
    
    # 计算平均值
    avg_psnr = total_psnr / count
    avg_uciqe = total_uciqe / count
    avg_uiqm = total_uiqm / count
    
    print("\nTest Set Results:")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average UCIQE: {avg_uciqe:.3f}")
    print(f"Average UIQM: {avg_uiqm:.3f}")
    
    return avg_psnr, avg_uciqe, avg_uiqm

def process_test_images(model, test_degraded_dir, test_clear_dir):
    """使用训练好的模型处理测试图像"""
    # 确保输出目录存在
    os.makedirs(test_clear_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 获取所有测试图像
    test_images = sorted(os.listdir(test_degraded_dir))
    total_images = len(test_images)
    
    # 创建转换器
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 逆转换，将tensor转回图像
    to_pil = transforms.ToPILImage()
    
    print(f"\n开始处理测试图像，共 {total_images} 张...")
    
    with torch.no_grad():
        for idx, img_name in enumerate(test_images):
            # 读取图像
            img_path = os.path.join(test_degraded_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            
            # 转换为tensor
            img_tensor = transform(img).unsqueeze(0).cuda()
            
            # 使用模型处理图像
            output = model(img_tensor)
            
            # 转换回图像格式
            output_img = output.squeeze(0).cpu()
            output_img = to_pil(output_img)
            
            # 保存处理后的图像
            save_path = os.path.join(test_clear_dir, img_name)
            output_img.save(save_path)
            
            # 显示进度
            print(f"已处理: {idx+1}/{total_images} - {img_name}")

def visualize_metrics(test_degraded_dir, test_clear_dir, model):
    """可视化测试集图像处理前后的评估指标"""
    # 初始化列表存储指标
    psnr_values = []
    uciqe_values = []
    uiqm_values = []
    image_names = []
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建转换器
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 获取所有测试图像
    test_images = sorted(os.listdir(test_degraded_dir))
    
    with torch.no_grad():
        for img_name in test_images:
            # 读取并处理图像
            img_path = os.path.join(test_degraded_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).cuda()
            
            # 使用模型处理图像
            output = model(img_tensor)
            
            # 转换为numpy数组
            output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
            
            # 计算指标
            clear_path = os.path.join(test_clear_dir, img_name)
            if os.path.exists(clear_path):
                clear_img = Image.open(clear_path).convert('RGB')
                clear_img = np.array(clear_img)
                psnr = calculate_psnr(clear_img, output_img)
                psnr_values.append(psnr)
            
            uciqe = calculate_uciqe(output_img)
            uiqm = calculate_uiqm(output_img)
            
            uciqe_values.append(uciqe)
            uiqm_values.append(uiqm)
            image_names.append(img_name)
    
    # 创建可视化图表
    plt.figure(figsize=(15, 10))
    
    # PSNR折线图
    if psnr_values:
        plt.subplot(311)
        plt.plot(psnr_values, 'b-', label='PSNR')
        plt.axhline(y=np.mean(psnr_values), color='r', linestyle='--', 
                   label=f'Average: {np.mean(psnr_values):.2f}')
        plt.title('PSNR Values')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # UCIQE折线图
    plt.subplot(312)
    plt.plot(uciqe_values, 'g-', label='UCIQE')
    plt.axhline(y=np.mean(uciqe_values), color='r', linestyle='--', 
                label=f'Average: {np.mean(uciqe_values):.3f}')
    plt.title('UCIQE Values')
    plt.xlabel('Image Index')
    plt.ylabel('UCIQE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # UIQM折线图
    plt.subplot(313)
    plt.plot(uiqm_values, 'm-', label='UIQM')
    plt.axhline(y=np.mean(uiqm_values), color='r', linestyle='--', 
                label=f'Average: {np.mean(uiqm_values):.3f}')
    plt.title('UIQM Values')
    plt.xlabel('Image Index')
    plt.ylabel('UIQM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_clear_dir, 'metrics_visualization.png'))
    plt.show()
    
    # 打印平均值
    print("\nTest Set Metrics Summary:")
    if psnr_values:
        print(f"Average PSNR: {np.mean(psnr_values):.2f}")
    print(f"Average UCIQE: {np.mean(uciqe_values):.3f}")
    print(f"Average UIQM: {np.mean(uiqm_values):.3f}")

# 修改主函数
if __name__ == "__main__":
    # 训练集路径
    train_degraded_dir = "data/train/degraded"
    train_clear_dir = "data/train/clear"
    
    # 测试集路径
    test_degraded_dir = "data/test/degraded"
    test_clear_dir = "data/test/clear"
    
    try:
        print("\n创建训练集数据加载器...")
        train_loader = create_dataloader(train_degraded_dir, train_clear_dir, batch_size=8, shuffle=True)
        
        # 训练模型
        model = train_model(train_loader, epochs=20, lr=1e-4)
        
        # 处理测试图像
        process_test_images(model, test_degraded_dir, test_clear_dir)
        visualize_metrics(test_degraded_dir, test_clear_dir, model)
        print("\n处理完成！处理后的图像已保存到:", test_clear_dir)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        # 建议检查步骤
        print("\n请检查:")
        print("1. 确保目录结构正确")
        print("2. 确保图像文件名匹配")
        print("3. 检查文件是否完整")
