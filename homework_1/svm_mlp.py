# 若欲使用 GPU，請在有 CUDA 的環境下執行；程式會自動偵測。

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------------- 設定（可調） ---------------------------
# 我把「完整收斂」相關的 epoch 設為比較高的數值；若想加速測試可調小。
SVM_EPOCHS = 300        # SVM 訓練 epochs（較多以達完全收斂）
MLP_EPOCHS = 800        # MLP 訓練 epochs（較多以達完全收斂）
BATCH_SIZE = 64
SVM_LR = 5e-3
MLP_LR = 5e-3
SVM_C = 1.0
MLP_L2 = 1e-4
LOG_EVERY = 20          # 每隔幾個 step 記錄一次 loss / Δw
VERBOSE_FIRST_N = 3     # 前幾個 step 印出細節（避免大量輸出）
SEED = 0

# --------------------------- 工具函式（中文說明） ---------------------------
def set_seed(seed=0):
    """設定 numpy 與 torch 的隨機種子，讓結果可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def make_blobs(n_samples=600, centers=2, dim=2, spread=1.2, seed=0, device='cpu', dtype=torch.float32):
    """
    產生分群資料（類似 sklearn.make_blobs）
    - centers: 群數（也就是類別數）
    - spread: 群內的標準差（越大越難分）
    回傳：
    - Xt: (N, dim) 的 torch.tensor（float）
    - yt: (N,) 的 torch.tensor（long），標籤 0..K-1
    註：使用 numpy 產生隨機，再轉為 torch tensor（簡單直觀）
    """
    rng = np.random.default_rng(seed)
    means = rng.uniform(-3, 3, size=(centers, dim))  # 每個 cluster 的中心位置隨機
    X, y = [], []
    per = n_samples // centers
    for i in range(centers):
        Xi = means[i] + spread * rng.standard_normal(size=(per, dim))
        X.append(Xi)
        y.append(np.full(per, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]
    Xt = torch.tensor(X, dtype=dtype, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    return Xt, yt

def one_hot_torch(y, K, device='cpu', dtype=torch.float32):
    """把長度為 N 的整數標籤 y（0..K-1）轉成 one-hot 矩陣 (N,K)"""
    Y = torch.zeros((y.size(0), K), dtype=dtype, device=device)
    Y[torch.arange(y.size(0), device=device), y] = 1.0
    return Y

# --------------------------- Linear SVM（手動更新） ---------------------------
class LinearSVMTrace:
    """
    教學用線性 SVM（使用 Hinge Loss + L2 regularization）
    注意：這裡使用逐筆更新（類似原始 numpy 版本的邏輯），不是使用 torch.optim。
    我保留這樣的好處是可以清楚觀察「當 margin >= 1 與 < 1 時」的不同更新規則。
    """

    def __init__(self, in_dim, C=1.0, lr=5e-3, epochs=10, batch_size=64,
                 log_every=10, verbose_first_n=5, device='cpu', dtype=torch.float32, seed=0):
        """
        主要參數說明：
        - in_dim: 輸入向量維度
        - C: Hinge loss 的權重（對錯分類懲罰程度）
        - lr: 學習率（影響更新幅度）
        - epochs, batch_size: 訓練迴圈控制
        - log_every, verbose_first_n: 記錄與顯示細節的頻率
        """
        self.C = C; self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.log_every = log_every; self.verbose_first_n = verbose_first_n
        self.device = device; self.dtype = dtype
        # 權重初始化為零（跟原始 numpy 範例一致）
        self.w = torch.zeros(in_dim, dtype=dtype, device=device)
        self.b = torch.tensor(0.0, dtype=dtype, device=device)
        # traces（用來畫學習曲線與觀察）
        self.w_norms = []     # 記錄 ‖w‖
        self.dw_norms = []    # 記錄單步 Δw 的範數
        self.losses = []      # 記錄 hinge+reg loss
        self.train_acc_hist = []
        self.test_acc_hist = []

    def hinge_loss_reg(self, w, b, X, y):
        """
        計算 Regularized Hinge Loss：
        L = 0.5 * ‖w‖^2 + C * mean( max(0, 1 - y*(Xw+b)) )
        注意：這裡假設 y 是 -1 或 +1
        """
        margins = 1 - y * (X.matmul(w) + b)      # (N,)
        hinge = torch.clamp(margins, min=0.0).mean()
        reg = 0.5 * torch.dot(w, w)
        return reg + self.C * hinge

    def fit(self, X, y, Xval=None, yval=None):
        """
        X: (N,d) float tensor
        y: (N,) long tensor in {0,1}  => 轉為 {-1,+1}
        Xval, yval: optional validation set
        """
        # map {0,1} -> {-1,+1}
        y2 = torch.where(y == 0,
                         torch.tensor(-1, dtype=self.dtype, device=self.device),
                         torch.tensor(1, dtype=self.dtype, device=self.device))
        n = X.size(0)
        step = 0
        for ep in range(self.epochs):
            # 隨機打亂（每個 epoch 產生新的 permutation）
            perm = torch.randperm(n, device=self.device)
            Xs = X[perm]; ys = y2[perm]

            # mini-batch（內層為逐筆更新以保留教學細節）
            for start in range(0, n, self.batch_size):
                Xe = Xs[start:start+self.batch_size]
                ye = ys[start:start+self.batch_size]
                if Xe.size(0) == 0: continue

                # 逐筆處理（為了模仿原始 numpy 版本的逐筆更新）
                for xi, yi in zip(Xe, ye):
                    margin = yi * (torch.dot(self.w, xi) + self.b)   # margin = y (w·x + b)
                    w_before = self.w.clone()

                    if margin >= 1.0:
                        # 若 margin >= 1，表示目前分類約束已滿足（不需 hinge 的梯度）
                        # 只是做 weight decay（等同於 L2 正則化梯度）
                        # w <- (1 - lr) * w
                        self.w = (1.0 - self.lr) * self.w
                        db = 0.0
                    else:
                        # 若 margin < 1，表示違反 margin，需要更新以減少錯誤
                        # w <- (1 - lr) w + lr * C * yi * xi
                        # b <- b + lr * C * yi
                        self.w = (1.0 - self.lr) * self.w + self.lr * self.C * yi * xi
                        db = self.lr * self.C * yi
                        self.b = self.b + db

                    dw = self.w - w_before   # Δw（可用來觀察更新量）

                    # 前幾步詳細列印，便於教學觀察
                    if step < self.verbose_first_n:
                        print(f"[SVM step {step}]")
                        print("  w(before)=", w_before.cpu().numpy())
                        print("  Δw       =", dw.cpu().numpy())
                        print("  w(after) =", self.w.cpu().numpy())
                        print("  ‖Δw‖     =", torch.norm(dw).item(), " ‖w‖=", torch.norm(self.w).item())

                    # 定期記錄 loss 與 ‖w‖、‖Δw‖
                    if step % self.log_every == 0:
                        L = self.hinge_loss_reg(self.w, self.b, X, y2)
                        self.losses.append(L.item())
                        self.w_norms.append(torch.norm(self.w).item())
                        self.dw_norms.append(torch.norm(dw).item())
                    step += 1

            # epoch 結尾記錄訓練/驗證準確率（方便觀察收斂）
            self.train_acc_hist.append(self.score(X, (y2 + 1) // 2))
            if Xval is not None and yval is not None:
                self.test_acc_hist.append(self.score(Xval, yval))

    def decision_function(self, X):
        """輸出連續分數，>0 表示 positive 類"""
        return X.matmul(self.w) + self.b

    def predict(self, X):
        """分類預測：decision >= 0 => 類別 1，否則 0"""
        return (self.decision_function(X) >= 0).long()

    def score(self, X, y):
        """回傳準確率（y in {0,1}）"""
        return (self.predict(X) == y).float().mean().item()

# --------------------------- MLP（單隱藏層，使用 autograd） ---------------------------
class MLPTrace(nn.Module):
    """
    單隱藏層 MLP（ReLU + Softmax）
    使用 autograd 計算梯度，但手動做參數更新（方便計算 ΔW 並觀察）
    Loss = CrossEntropy + L2 regularization
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 lr=5e-3, epochs=60, batch_size=64, l2=1e-4,
                 log_every=10, verbose_first_n=5, device='cpu', seed=0):
        super().__init__()
        self.device = device
        torch.manual_seed(seed)
        # 使用 nn.Linear 實作兩層（包含 bias）
        # 注意：nn.Linear 的 weight 形狀為 (out_features, in_features)
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        # 使用 He init（kaiming）來初始化權重，對 ReLU 常有效
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size; self.l2 = l2
        self.log_every = log_every; self.verbose_first_n = verbose_first_n

        # traces
        self.losses = []
        self.dw1_norms = []
        self.dw2_norms = []
        self.train_acc_hist = []
        self.test_acc_hist = []

    def forward(self, x):
        """
        forward pass（回傳 logits，不做 softmax）
        z1 = X W1^T + b1
        h  = ReLU(z1)
        z2 = h W2^T + b2
        output = logits z2
        """
        h = torch.relu(self.fc1(x))
        z = self.fc2(h)
        return z

    def ce_loss(self, logits, y_onehot):
        """
        計算 cross-entropy + L2 regularization 的損失
        - logits: (B,K)
        - y_onehot: (B,K)
        我這裡用 log_softmax 做數值穩定處理。
        """
        logp = torch.log_softmax(logits, dim=1)
        ce = - (y_onehot * logp).sum(dim=1).mean()  # batch 平均
        reg = 0.5 * self.l2 * (torch.sum(self.fc1.weight**2) + torch.sum(self.fc2.weight**2))
        return ce + reg

    def fit(self, X, y, Xval=None, yval=None):
        """
        訓練迴圈：
        - 使用 autograd 計算 grads（loss.backward()）
        - 使用 torch.no_grad() 手動更新參數（以便在更新前後計算 ΔW）
        - 記錄 losses、‖ΔW1‖、‖ΔW2‖、train/test accuracy
        """
        n = X.size(0)
        K = int(y.max().item()) + 1
        Y = one_hot_torch(y, K, device=self.device)
        step = 0
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            Xs = X[perm]; Ys = Y[perm]
            for st in range(0, n, self.batch_size):
                Xe = Xs[st:st+self.batch_size]
                Ye = Ys[st:st+self.batch_size]
                if Xe.size(0) == 0: continue

                # forward
                logits = self.forward(Xe)                # (B,K)
                P = torch.softmax(logits, dim=1)        # (B,K)（僅供觀察/紀錄用）
                loss = self.ce_loss(logits, Ye)

                # backward
                self.zero_grad()
                loss.backward()

                # 保存更新前的權重（detach，使其不會連回計算圖）
                W1_before = self.fc1.weight.detach().clone()
                W2_before = self.fc2.weight.detach().clone()

                # 手動更新（SGD），同時加入 L2（weight decay 的直接寫法）
                with torch.no_grad():
                    # 注意 grad 的形狀與 scale 已由 backward 處理（loss 是 batch-mean）
                    self.fc2.weight -= self.lr * (self.fc2.weight.grad + self.l2 * self.fc2.weight)
                    self.fc2.bias   -= self.lr * self.fc2.bias.grad
                    self.fc1.weight -= self.lr * (self.fc1.weight.grad + self.l2 * self.fc1.weight)
                    self.fc1.bias   -= self.lr * self.fc1.bias.grad

                # 計算 ΔW（用來觀察更新量）
                dW1 = (self.fc1.weight.detach() - W1_before)
                dW2 = (self.fc2.weight.detach() - W2_before)

                # 前幾步印出 ΔW 的範數，做教學展示
                if step < self.verbose_first_n:
                    print(f"[MLP step {step}]")
                    print("  ‖ΔW1‖=", torch.norm(dW1).item(), " ‖ΔW2‖=", torch.norm(dW2).item())

                # 定期記錄 loss 與 ΔW 的範數
                if step % self.log_every == 0:
                    self.losses.append(loss.item())
                    self.dw1_norms.append(torch.norm(dW1).item())
                    self.dw2_norms.append(torch.norm(dW2).item())
                step += 1

            # epoch 結束紀錄準確率
            self.train_acc_hist.append(self.score(X, y))
            if Xval is not None and yval is not None:
                self.test_acc_hist.append(self.score(Xval, yval))

    def predict(self, X):
        logits = self.forward(X)
        return torch.argmax(logits, dim=1)

    def score(self, X, y):
        return (self.predict(X) == y).float().mean().item()

# --------------------------- 主程式：建立資料、訓練、繪圖 ---------------------------
def main():
    # 設定隨機種子
    set_seed(SEED)

    # 選擇裝置（GPU 如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # ---------------- SVM：二類資料（線性可分，方便觀察決策邊界） ----------------
    # 為了教學方便，我使用 2D 的資料（可視化）
    X2, y2 = make_blobs(n_samples=600, centers=2, dim=2, spread=1.0, seed=42, device=device)
    split = int(0.8 * len(X2))
    Xtr2, Xte2 = X2[:split], X2[split:]
    ytr2, yte2 = y2[:split], y2[split:]

    print("\n=== Training Linear SVM (完整收斂設定) ===")
    svm = LinearSVMTrace(in_dim=2, C=SVM_C, lr=SVM_LR, epochs=SVM_EPOCHS,
                         batch_size=BATCH_SIZE, log_every=LOG_EVERY, verbose_first_n=VERBOSE_FIRST_N,
                         device=device, dtype=torch.float32, seed=SEED)
    svm.fit(Xtr2, ytr2, Xte2, yte2)
    print("SVM train acc:", svm.score(Xtr2, ytr2))
    print("SVM test  acc:", svm.score(Xte2, yte2))

    # ---------------- MLP：三類資料（較適合觀察非線性邊界） ----------------
    X3, y3 = make_blobs(n_samples=900, centers=3, dim=2, spread=1.3, seed=7, device=device)
    split = int(0.8 * len(X3))
    Xtr3, Xte3 = X3[:split], X3[split:]
    ytr3, yte3 = y3[:split], y3[split:]

    print("\n=== Training MLP (完整收斂設定) ===")
    mlp = MLPTrace(in_dim=2, hidden_dim=32, out_dim=3, lr=MLP_LR, epochs=MLP_EPOCHS,
                   batch_size=BATCH_SIZE, l2=MLP_L2, log_every=LOG_EVERY,
                   verbose_first_n=VERBOSE_FIRST_N, device=device, seed=SEED)
    mlp.to(device)
    mlp.fit(Xtr3, ytr3, Xte3, yte3)
    print("MLP train acc:", mlp.score(Xtr3, ytr3))
    print("MLP test  acc:", mlp.score(Xte3, yte3))

    # ------------------- 繪圖（把 traces 移到 CPU，再用 matplotlib） -------------------
    # 為了避免 matplotlib 與 CUDA 的相容性問題，先把 list 資料拿回 CPU（它們已是 Python float）
    # SVM 繪圖
    plt.figure(figsize=(6,4))
    plt.plot(svm.losses)
    plt.title("SVM：Regularized Hinge Loss（訓練過程）", fontsize=12)
    plt.xlabel(f"record step (每 {LOG_EVERY} 個更新記錄一次)", fontsize=10)
    plt.ylabel("loss", fontsize=10)
    plt.grid(True)

    plt.figure(figsize=(6,4))
    plt.plot(svm.w_norms, label="‖w‖")
    plt.plot(svm.dw_norms, label="‖Δw‖")
    plt.title("SVM：‖w‖ 與 ‖Δw‖（隨步數變化）", fontsize=12)
    plt.xlabel(f"record step (每 {LOG_EVERY} 個更新記錄一次)", fontsize=10)
    plt.legend(); plt.grid(True)

    plt.figure(figsize=(6,4))
    plt.plot(svm.train_acc_hist, label="train")
    if len(svm.test_acc_hist) > 0:
        plt.plot(svm.test_acc_hist, label="test")
    plt.title("SVM：每 epoch 的準確率（Accuracy）", fontsize=12)
    plt.xlabel("epoch", fontsize=10); plt.ylabel("acc", fontsize=10)
    plt.legend(); plt.grid(True)

    # MLP 繪圖
    plt.figure(figsize=(6,4))
    plt.plot(mlp.losses)
    plt.title("MLP：Cross-Entropy Loss（訓練過程）", fontsize=12)
    plt.xlabel(f"record step (每 {LOG_EVERY} 個更新記錄一次)", fontsize=10)
    plt.ylabel("loss", fontsize=10)
    plt.grid(True)

    plt.figure(figsize=(6,4))
    plt.plot(mlp.dw1_norms, label="‖ΔW1‖")
    plt.plot(mlp.dw2_norms, label="‖ΔW2‖")
    plt.title("MLP：權重更新量範數（‖ΔW1‖、‖ΔW2‖）", fontsize=12)
    plt.xlabel(f"record step (每 {LOG_EVERY} 個更新記錄一次)", fontsize=10)
    plt.legend(); plt.grid(True)

    plt.figure(figsize=(6,4))
    plt.plot(mlp.train_acc_hist, label="train")
    if len(mlp.test_acc_hist) > 0:
        plt.plot(mlp.test_acc_hist, label="test")
    plt.title("MLP：每 epoch 的準確率（Accuracy）", fontsize=12)
    plt.xlabel("epoch", fontsize=10); plt.ylabel("acc", fontsize=10)
    plt.legend(); plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
