import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from bottleneck_transformer_pytorch import BottleStack
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy.stats import skew, kurtosis




def init_file_and_num(number):
    number = number
    folder_path = f"bottleneck-transformer-pytorch-main\\results\\demo{number}"  #plt.savefig本身不带有创建文件夹的功能 必须手动创建
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# 编译正则表达式，匹配文件名前缀：一个汉字后跟一个或多个数字
prefix_pattern = re.compile(r'^.*$')

# 自定义数据集类
class RockDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def extract_handcrafted_features(self, image):
        image_np = np.array(image)
        image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, C)
    
        # ================= 颜色特征 =================
        # 转换到HSV空间（网页1[1,2,3](@ref)）
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
        # HSV直方图（32维）
        h_hist = np.histogram(hsv_image[...,0], bins=16, range=(0,180))[0]  # H分16bin（网页3[3](@ref)）
        s_hist = np.histogram(hsv_image[...,1], bins=8, range=(0,256))[0]   # S分8bin
        v_hist = np.histogram(hsv_image[...,2], bins=8, range=(0,256))[0]    # V分8bin
        hsv_features = np.concatenate([h_hist, s_hist, v_hist])  # 32维
    
        # 颜色矩（9维，网页4[4,5](@ref)）
        color_moments = []
        for i in range(3):  # 对H/S/V三个通道
            channel = hsv_image[...,i].flatten()
            color_moments.extend([
                np.mean(channel),          # 一阶矩（均值）
                np.std(channel),           # 二阶矩（标准差）
                skew(channel)              # 三阶矩（偏度）
            ])
        color_moments = np.array(color_moments)  # 9维
    
        # ================= 亮度特征 =================（网页6[6,7](@ref)）
        gray_image = rgb2gray(image_np) * 255
        gray_image = gray_image.astype(np.uint8)
    
        # 统计特征（4维）
        brightness_features = [
            np.mean(gray_image),          # 平均亮度
            np.std(gray_image),           # 亮度对比度
            skew(gray_image.flatten()),   # 亮度分布对称性
            kurtosis(gray_image.flatten())# 亮度峰态
        ]
    
        # ================= 纹理特征 =================（网页9[9,10,11](@ref)）
        # GLCM特征（32维）
        glcm = graycomatrix(gray_image, 
                            distances=[1, 3], 
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        glcm_features = []
        for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
            glcm_features.extend(graycoprops(glcm, prop).flatten())  # 4属性×4角度×2距离=32维
    
        # LBP特征（32维，网页10[10](@ref)）
        lbp_radius = 2
        lbp_points = 16
        lbp = local_binary_pattern(gray_image, 
                              P=lbp_points, 
                              R=lbp_radius, 
                              method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=lbp_points*2, range=(0, lbp_points+2))
    
        # Gabor滤波特征（32维，网页9[9](@ref)）
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi/4):  # 4个方向
            for sigma in [1.0, 3.0]:                # 2个尺度
                filt_real, filt_imag = gabor(gray_image, 
                                       frequency=0.6/sigma, 
                                       theta=theta,
                                       sigma_x=sigma, 
                                       sigma_y=sigma)
                gabor_features.extend([
                    np.mean(filt_real),  # 实部均值
                    np.std(filt_real),    # 实部方差
                    np.mean(filt_imag),   # 虚部均值
                    np.std(filt_imag)     # 虚部方差
                ] )  # 4方向×2尺度×4统计量=32维
    
        # ================= 特征融合 =================
        all_features = np.concatenate([
            hsv_features,               # 32维
            color_moments,              # 9维  
            brightness_features,        # 4维
            glcm_features,              # 32维
            lbp_hist,                   # 32维
            gabor_features              # 32维
        ])  # 总维度32+9+4+32+32+32=141
    
        # 归一化处理（网页3[3](@ref)）
        all_features = (all_features - np.mean(all_features)) / (np.std(all_features) + 1e-6)
    
        return torch.tensor(all_features, dtype=torch.float32)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # 提取手工特征
        handcrafted_features = self.extract_handcrafted_features(image)
        
        return image, handcrafted_features, label


# 定义 MLP 类
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        # 特征通过多层非线性变换
        x = self.fc(x)  # 维度变化: [batch, input_dim] → [batch, output_dim]
        return x    


# 构造 BotNet 模型（利用 ResNet50 的前5个子模块 + BottleStack 模块）
class BotNet(nn.Module):
    def __init__(self, num_classes, handcrafted_feature_dim, hidden_dim, freeze_resnet_layers):
        super(BotNet, self).__init__()
        # 加载预训练的 ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        backbone = list(resnet.children())[:5]  # 提取前5层
        
        # 冻结 ResNet50 的前 freeze_resnet_layers 层
        for param in list(resnet.parameters())[:freeze_resnet_layers]:
            param.requires_grad = False
        
        # BottleStack 模块
        bot_layer = BottleStack(
            dim=256,
            fmap_size=56,
            dim_out=2048,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=nn.ReLU()
        )

        # 拼接手工特征和卷积输出特征
        self.model = nn.Sequential(
            *backbone,
            bot_layer,
            nn.AdaptiveAvgPool2d((1, 1)),  # 池化
            nn.Flatten(1)
        )

        # 全连接层之前需要拼接手工特征的维度
        self.fc_input_dim = 2048 + handcrafted_feature_dim  # 2048是卷积输出的通道数

        # 使用 MLP 替代单一的全连接层
        self.mlp = MLP(input_dim=self.fc_input_dim, hidden_dim=hidden_dim, output_dim=num_classes)

    def forward(self, x, handcrafted_features):
        # 获取卷积层的输出
        x = self.model(x)

        # 拼接卷积输出和手工特征
        x = torch.cat((x, handcrafted_features), dim=1)  # 在特征维度拼接
        
        # 使用 MLP 进行分类
        x = self.mlp(x)
        
        return x



def main():
    writer = SummaryWriter(log_dir=f"bottleneck-transformer-pytorch-main\\runs\\experiment_{number}")

    # 修正后的训练集增强流程（调整RandomAffine位置）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.005),
        transforms.ToTensor(),
        ])

    # 修正后的验证集增强流程（使用中心裁剪）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_dir = "bottleneck-transformer-pytorch-main\dataset"
    # 使用 ImageFolder 加载数据集，要求数据集目录下每个子文件夹名称代表一个类别
    full_dataset = datasets.ImageFolder(root=dataset_dir)

    # 存储所有图片的路径和标签
    image_paths = []  # 保存每张图片的完整路径
    labels = []       # 保存每张图片对应的标签

    # 替换为直接划分全部图片索引：
    all_indices = list(range(len(full_dataset.imgs)))
    print("总图片数:", len(all_indices))

    # 填充 image_paths 和 labels
    image_paths = [img_path for img_path, _ in full_dataset.imgs]  # 提取所有图片路径
    labels = [label for _, label in full_dataset.imgs]             # 提取所有标签

    train_indices, val_indices = train_test_split(
        all_indices, 
        test_size=0.2, 
        random_state=42,
    )

    # 创建数据集时直接使用索引
    train_dataset = RockDataset(
        image_paths=[image_paths[i] for i in train_indices],
        labels=[labels[i] for i in train_indices],
        transform=train_transform
    )
    val_dataset = RockDataset(
        image_paths=[image_paths[i] for i in val_indices],
        labels=[labels[i] for i in val_indices],
        transform=val_transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=os.cpu_count()-1,  # 使用全部CPU核心[4,6](@ref)
        pin_memory=True,               # 启用内存锁页加速传输[4](@ref)
        prefetch_factor=2,             # 预加载2个batch[7](@ref)
        persistent_workers=True,        # 保持worker进程存活
        drop_last = True
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=os.cpu_count()-1,
        pin_memory=True,    
        prefetch_factor=2,  
        persistent_workers=True,   
        drop_last = True)

    num_classes = len(full_dataset.classes)
    print("分类的类别数:", num_classes)
    print("类别名称:", full_dataset.classes)

    # 构造 BotNet 模型
    model = BotNet(num_classes=num_classes, handcrafted_feature_dim=141, hidden_dim=128, freeze_resnet_layers=0)  # 冻结前5层
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    class_weights = torch.tensor([1.0, 1.0], device=device)  # 尝试权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 修改损失函数定义
    optimizer_groups = [
        # 第一组：ResNet主干参数（网页1[1](@ref)）
        {'params': [p for n,p in model.named_parameters() 
                if 'model' in n and 'bn' not in n and p.requires_grad], 'lr': 0.0001},
        # 第二组：MLP全连接层（网页6[6](@ref)）
        {'params': model.mlp.parameters(), 'lr': 0.0003},
        # 第三组：BatchNorm参数（网页3[3](@ref)）
        {'params': [p for n,p in model.named_parameters() 
                if 'bn' in n], 'lr': 0.0001, 'weight_decay': 0}
        ]
    optimizer = optim.AdamW(optimizer_groups)
    accumulation_steps = 1

    # 训练与验证循环
    num_epochs = 10
    
    # 创建1Cycle策略学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[0.0001, 0.0003, 0.0001],  # 对应三个参数组的初始学习率
        pct_start=0.15,                # 学习率上升阶段占10%（网页4[4](@ref)）
        total_steps=num_epochs,
        anneal_strategy='linear',      # 线性退火（网页2[2](@ref)）
        div_factor=25.0,               # 初始学习率= max_lr/div_factor（网页1[1](@ref)）
        final_div_factor=1e4            # 最终学习率=初始学习率/final_div_factor
        )

    # 在训练循环中添加（需CUDA支持）
    scaler = torch.amp.GradScaler('cuda',
                                  init_scale=16384.0,
                                  growth_factor=2,
                                  backoff_factor=0.4,
                                  growth_interval=100
                                  )
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, (images, handcrafted_features, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            images = images.to(device)
            handcrafted_features = handcrafted_features.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            # 前向计算（自动混合精度）
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(images, handcrafted_features)
                loss = criterion(outputs, labels)  # 不再手动除以 accumulation_steps
    
            # 反向传播（自动缩放梯度）
            scaler.scale(loss).backward()
    
            # 梯度累积控制
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 可选：梯度裁剪（需先解除缩放）
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
                # 更新参数及缩放因子
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
                # 监控缩放因子
                current_scale = scaler.get_scale()
                writer.add_scalar('AMP/scale_factor', current_scale, epoch)
    
            # 统计训练损失和准确率
            running_loss += loss.item() * labels.size(0)  # 修正：无需乘以 accumulation_steps
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        #训练集loss与学习率
        epoch_loss = running_loss / total_train
        train_acc = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Training Accuracy = {train_acc:.2f}%")
        
        #ResNet梯度幅值 对比 MLP梯度幅值 （用于调试）
        #resnet_grad = torch.norm(torch.cat([p.grad.flatten() 
        #            for p in model.model.parameters()]))
        #mlp_grad = torch.norm(torch.cat([p.grad.flatten() 
        #            for p in model.mlp.parameters()]))
        #print(f"ResNet梯度幅值: {resnet_grad:.4f}, MLP梯度幅值: {mlp_grad:.4f}")

        # 将训练损失和准确率记录到 TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('grad/train_resnet_grad', resnet_grad, epoch)
        #writer.add_scalar('grad/train_mlp_grad', mlp_grad, epoch)

        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, handcrafted_features, labels in tqdm(val_loader, desc="Validation"):
                images, handcrafted_features, labels = images.to(device), handcrafted_features.to(device), labels.to(device)
                outputs = model(images, handcrafted_features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / total_val
        val_acc = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # 将验证损失和准确率记录到 TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 计算并显示混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        save_dir_plt = f"bottleneck-transformer-pytorch-main\\results\\demo{number}"+f"\\confusion_matrix_epoch_{epoch+1}.png"
        plt.savefig(save_dir_plt, bbox_inches='tight', dpi=300)
        plt.clf()

        # 将学习率记录到 TensorBoard
        writer.add_scalar('LR/resnet', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/mlp', optimizer.param_groups[1]['lr'], epoch)
        writer.add_scalar('LR/bn', optimizer.param_groups[2]['lr'], epoch)

        # 更新学习率
        scheduler.step()    

        # 保存模型
        save_model_dir = f"bottleneck-transformer-pytorch-main\\results\\demo{number}"+f"\\botnet_model{number}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_model_dir)

    # 关闭 TensorBoard 的 writer
    writer.close()


if __name__ == '__main__':
    import multiprocessing
    number = 3  # 设置实验编号
    init_file_and_num(number)  # 初始化文件夹
    multiprocessing.freeze_support()
    main()