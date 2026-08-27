import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"检测到设备: {device}")


def euler_to_cartesian(pitch, yaw):
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    return np.stack([x, y, z], axis=-1)


class OrthodromicLoss(nn.Module):
    def __init__(self):
        super(OrthodromicLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=-1)
        y_true = torch.nn.functional.normalize(y_true, p=2, dim=-1)
        dot_product = torch.sum(y_pred * y_true, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(dot_product)
        return torch.mean(angle)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiModalTransformer(nn.Module):
    def __init__(self, traj_dim=3, vis_dim=15, d_model=128, nhead=4, num_layers=4, seq_len_in=20, seq_len_out=10):
        super(MultiModalTransformer, self).__init__()
        self.seq_len_out = seq_len_out
        self.traj_dim = traj_dim

        combined_dim = traj_dim + vis_dim
        self.input_projection = nn.Linear(combined_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(seq_len_in * d_model, seq_len_out * traj_dim)

    def forward(self, traj_src, vis_src):
        x = torch.cat((traj_src, vis_src), dim=-1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        out = self.output_layer(x)
        out = out.view(-1, self.seq_len_out, self.traj_dim)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out


def load_multimodal_data(traj_folder, vis_folder, seq_len_in=20, seq_len_out=10):
    print(f"正在进行时空对齐...\n 轨迹路径: {traj_folder}\n 视觉路径: {vis_folder}")
    X_traj_all, X_vis_all, Y_all = [], [], []
    files = sorted([f for f in os.listdir(traj_folder) if f.endswith('.txt')])

    for file_name in files:
        vid_id = file_name.split('.')[0]
        traj_path = os.path.join(traj_folder, file_name)
        with open(traj_path, 'r') as f:
            lines = f.readlines()
        for i in range(1, len(lines) - 1, 2):
            try:
                pitch_list = [float(x) for x in lines[i].strip().split()]
                yaw_list = [float(x) for x in lines[i + 1].strip().split()]
                traj_3d = euler_to_cartesian(np.array(pitch_list), np.array(yaw_list))
                total_len = len(traj_3d)
                step = 1
                for j in range(0, total_len - seq_len_in - seq_len_out, step):
                    current_traj_x = traj_3d[j: j + seq_len_in]
                    current_traj_y = traj_3d[j + seq_len_in: j + seq_len_in + seq_len_out]
                    start_time_sec = j * 0.1
                    chunk_start = int(start_time_sec // 2) * 2
                    chunk_end = chunk_start + 2
                    chunk_folder_name = f"{chunk_start}-{chunk_end}s"
                    vis_file_path = os.path.join(vis_folder, chunk_folder_name, f"{vid_id}.txt")
                    current_vis_x = np.zeros((seq_len_in, 15))
                    if os.path.exists(vis_file_path):
                        with open(vis_file_path, 'r') as vf:
                            v_lines = vf.readlines()
                            read_len = min(seq_len_in, len(v_lines))
                            for k in range(read_len):
                                parts = v_lines[k].strip().split()
                                if len(parts) >= 17:
                                    current_vis_x[k] = [float(p) for p in parts[2:17]]
                    X_traj_all.append(current_traj_x)
                    X_vis_all.append(current_vis_x)
                    Y_all.append(current_traj_y)
            except Exception as e:
                continue

    X_traj_tensor = torch.tensor(np.array(X_traj_all), dtype=torch.float32)
    X_vis_tensor = torch.tensor(np.array(X_vis_all), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_all), dtype=torch.float32)
    print(f"✅ 时空对齐完成！样本总数: {len(X_traj_tensor)}")
    return X_traj_tensor, X_vis_tensor, Y_tensor


def split_train_test(X_traj, X_vis, Y, test_ratio=0.2, seed=42):
    n = len(X_traj)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    split = int(n * (1 - test_ratio))
    train_idx = perm[:split]
    test_idx = perm[split:]
    return (X_traj[train_idx], X_vis[train_idx], Y[train_idx],
            X_traj[test_idx], X_vis[test_idx], Y[test_idx])


def evaluate_on_set(model, X_traj, X_vis, Y, device, batch_size=256):
    model.eval()
    total_loss = 0.0
    n = len(X_traj)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_X_traj = X_traj[i:i + batch_size].to(device)
            batch_X_vis = X_vis[i:i + batch_size].to(device)
            batch_Y = Y[i:i + batch_size].to(device)
            preds = model(batch_X_traj, batch_X_vis)
            preds_n = torch.nn.functional.normalize(preds, p=2, dim=-1)
            Y_n = torch.nn.functional.normalize(batch_Y, p=2, dim=-1)
            dot = torch.sum(preds_n * Y_n, dim=-1)
            dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
            angle = torch.acos(dot).mean()
            total_loss += angle.item() * (batch_X_traj.size(0))
    return total_loss / n


def evaluate_and_visualize(model, X_traj_tensor, X_vis_tensor, Y_tensor, sample_idx=0):
    model.eval()
    print(f"\n正在生成第 {sample_idx} 个样本的预测对比图...")
    with torch.no_grad():
        x_traj_input = X_traj_tensor[sample_idx:sample_idx + 1].to(device)
        x_vis_input = X_vis_tensor[sample_idx:sample_idx + 1].to(device)
        y_true = Y_tensor[sample_idx].numpy()
        y_pred = model(x_traj_input, x_vis_input).squeeze(0).cpu().numpy()

    true_z = y_true[:, 2]
    pred_z = y_pred[:, 2]
    past_z = X_traj_tensor[sample_idx].numpy()[:, 2]

    plt.figure(figsize=(8, 5))
    time_past = np.arange(-len(past_z), 0)
    plt.plot(time_past, past_z, label="Past Trajectory (2.0s)", color='gray', linestyle='--')
    time_future = np.arange(0, len(true_z))
    plt.plot(time_future, true_z, label="Ground Truth (1.0s)", color='green', marker='o')
    plt.plot(time_future, pred_z, label="MMT Prediction (1.0s)", color='red', marker='x')
    plt.axvline(x=0, color='black', linestyle=':')
    plt.title(f"MMT Prediction (test sample #{sample_idx})")
    plt.xlabel("Time step (0.1s)")
    plt.ylabel("Normalized Z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    TRAIN_MODE = True

    model_save_path = "multimodal_transformer_weights.pth"
    seq_in, seq_out = 20, 10

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "AggregatedDataset")
    vis_features_path = os.path.join(BASE_DIR, "ExtractedFeatures")

    cache_file = os.path.join(BASE_DIR, "X_traj_full.pt")

    if os.path.exists(cache_file):
        print("📂 发现本地缓存，正在直接读取已保存的全量数据...")
        X_traj_data = torch.load(os.path.join(BASE_DIR, "X_traj_full.pt"))
        X_vis_data = torch.load(os.path.join(BASE_DIR, "X_vis_full.pt"))
        Y_data = torch.load(os.path.join(BASE_DIR, "Y_full.pt"))
    else:
        print("⏳ 未发现本地缓存，开始首次全量数据提取与时空对齐...")
        X_traj_data, X_vis_data, Y_data = load_multimodal_data(dataset_path, vis_features_path, seq_in, seq_out)
        torch.save(X_traj_data, os.path.join(BASE_DIR, "X_traj_full.pt"))
        torch.save(X_vis_data, os.path.join(BASE_DIR, "X_vis_full.pt"))
        torch.save(Y_data, os.path.join(BASE_DIR, "Y_full.pt"))

    X_traj_tr, X_vis_tr, Y_tr, X_traj_te, X_vis_te, Y_te = split_train_test(
        X_traj_data, X_vis_data, Y_data, test_ratio=0.2, seed=42
    )
    print(f"训练样本: {len(X_traj_tr)}, 测试样本: {len(X_traj_te)}")

    model = MultiModalTransformer(traj_dim=3, vis_dim=15, d_model=128, nhead=4, num_layers=4,
                                  seq_len_in=20, seq_len_out=10).to(device)

    if TRAIN_MODE:
        print("====== 进入【训练模式】 ======")
        criterion = OrthodromicLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 40
        batch_size = 256
        total_batches = math.ceil(len(X_traj_tr) / batch_size)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for i in range(0, len(X_traj_tr), batch_size):
                batch_X_traj = X_traj_tr[i:i + batch_size].to(device)
                batch_X_vis = X_vis_tr[i:i + batch_size].to(device)
                batch_Y = Y_tr[i:i + batch_size].to(device)

                predictions = model(batch_X_traj, batch_X_vis)
                loss = criterion(predictions, batch_Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_gcd = evaluate_on_set(model, X_traj_tr, X_vis_tr, Y_tr, device, batch_size)
            test_gcd = evaluate_on_set(model, X_traj_te, X_vis_te, Y_te, device, batch_size)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch +1:02d}/{epochs}] "
 f"Train GCD: {train_gcd:.4f} rad | "
                      f"Test GCD: {test_gcd:.4f} rad")

        torch.save(model.state_dict(), model_save_path)
        print(f"✅ 模型权重保存至 {model_save_path}")

    else:
        print("====== 进入【测试/画图模式】 ======")
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print("❌ 找不到模型权重文件")
            exit()

    if len(X_traj_te) >= 200:
        evaluate_and_visualize(model, X_traj_te, X_vis_te, Y_te, sample_idx=10)
        evaluate_and_visualize(model, X_traj_te, X_vis_te, Y_te, sample_idx=50)
        evaluate_and_visualize(model, X_traj_te, X_vis_te, Y_te, sample_idx=100)
    else:
        evaluate_and_visualize(model, X_traj_te, X_vis_te, Y_te, sample_idx=0)
