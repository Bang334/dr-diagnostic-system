"""
PyTorch Implementation and Training of Prototypical Networks for Few-shot Diabetic Retinopathy Grading.
Research scaffold migrated into dr-diagnostic-system.
All images in this demo are simulated tensors; reported accuracy is not a
medical-model result and must not be included as clinical evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. BỘ TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTOR BACKBONE)
class FundusFeatureExtractor(nn.Module):
    """Backbone mạng CNN nhỏ để trích xuất vector đặc trưng từ ảnh võng mạc đáy mắt."""
    def __init__(self, embedding_dim=16):
        super(FundusFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 8 x 112 x 112
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 16 x 56 x 56
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: 32 x 1 x 1
        )
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        embeddings = self.fc(features)
        return embeddings


# 2. MẠNG NGUYÊN MẪU (PROTOTYPICAL NETWORKS)
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    @staticmethod
    def compute_distances(query_embeddings, prototypes):
        """Tính toán khoảng cách Euclide giữa các vector truy vấn và các Prototype nguyên mẫu lớp y khoa."""
        n_query = query_embeddings.size(0)
        n_class = prototypes.size(0)
        
        query_expand = query_embeddings.unsqueeze(1).expand(n_query, n_class, -1)
        proto_expand = prototypes.unsqueeze(0).expand(n_query, n_class, -1)
        
        # Khoảng cách Euclide bình phương
        distances = torch.pow(query_expand - proto_expand, 2).sum(dim=2)
        return distances # [num_queries, num_classes]

    def forward(self, support_images, support_labels, query_images):
        """
        support_images: Dữ liệu tập hỗ trợ (support set) chứa ít ảnh có nhãn
        support_labels: Nhãn của tập hỗ trợ (0 đến 4)
        query_images: Dữ liệu tập truy vấn (query set) cần dự đoán nhãn
        """
        support_embeddings = self.backbone(support_images) # [num_supports, embedding_dim]
        query_embeddings = self.backbone(query_images)     # [num_queries, embedding_dim]
        
        # Lấy danh sách các lớp duy nhất (theo đúng thứ tự tăng dần)
        unique_classes = torch.unique(support_labels)
        prototypes = []
        for c in unique_classes:
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
            
        prototypes = torch.stack(prototypes) # [num_classes, embedding_dim]
        
        distances = self.compute_distances(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        return log_p_y


# 3. HÀM GIẢ LẬP DỮ LIỆU EPISODE CHO FEW-SHOT
def generate_simulated_episode(num_classes=5, num_support=2, num_query=3, img_shape=(3, 224, 224)):
    """
    Sinh dữ liệu giả lập cho 1 tập episode.
    Để mô hình có thể học được đặc trưng, mỗi lớp c sẽ được cộng thêm một bias đặc trưng riêng.
    """
    support_imgs = []
    support_lbls = []
    for c in range(num_classes):
        for _ in range(num_support):
            # Mỗi lớp c có đặc trưng phân biệt dạng bias
            img = torch.randn(*img_shape) + c * 0.4
            support_imgs.append(img)
            support_lbls.append(c)
            
    query_imgs = []
    query_lbls = []
    for c in range(num_classes):
        for _ in range(num_query):
            img = torch.randn(*img_shape) + c * 0.4
            query_imgs.append(img)
            query_lbls.append(c)
            
    return (
        torch.stack(support_imgs),
        torch.tensor(support_lbls),
        torch.stack(query_imgs),
        torch.tensor(query_lbls)
    )


# 4. CHƯƠNG TRÌNH HUẤN LUYỆN EPISODIC TRAINING
def run_few_shot_training_demo():
    print("--> Starting simulated Few-Shot Episodic Training (ProtoNet)...")
    
    # 5-way Few-shot configuration
    num_classes = 5
    num_support = 2 # 2-shot learning
    num_query = 3   # 3 queries per class
    embedding_dim = 16
    
    backbone = FundusFeatureExtractor(embedding_dim=embedding_dim)
    protonet = PrototypicalNetworks(backbone)
    optimizer = optim.Adam(protonet.parameters(), lr=0.001)
    
    print(f"   Config: {num_classes}-way {num_support}-shot learning scenario.")
    print("--> Training ProtoNet for 10 epochs on simulated medical images...")
    
    # Chạy vòng lặp huấn luyện (Training Loop)
    for epoch in range(1, 11):
        protonet.train()
        optimizer.zero_grad()
        
        # Sinh tập dữ liệu huấn luyện y khoa giả lập cho Episode hiện tại
        support_imgs, support_lbls, query_imgs, query_lbls = generate_simulated_episode(
            num_classes=num_classes, num_support=num_support, num_query=num_query
        )
        
        # Forward pass qua mạng ProtoNet
        log_probabilities = protonet(support_imgs, support_lbls, query_imgs)
        
        # Tính toán loss bằng Negative Log Likelihood Loss
        loss = F.nll_loss(log_probabilities, query_lbls)
        
        # Backpropagation & Cập nhật tham số mô hình
        loss.backward()
        optimizer.step()
        
        # Tính độ chính xác dự đoán (Accuracy)
        predictions = torch.argmax(log_probabilities, dim=1)
        acc = (predictions == query_lbls).float().mean()
        
        print(f"   Epoch {epoch:02d}/10 - Loss: {loss.item():.4f} - Accuracy: {acc.item()*100:.2f}%")
        
    print("\n--> Training completed successfully. Let's test the trained model on a new Query Set:")
    
    # Chạy thử nghiệm mô hình đã train trên tập test (Inference)
    protonet.eval()
    with torch.no_grad():
        test_support_imgs, test_support_lbls, test_query_imgs, test_query_lbls = generate_simulated_episode(
            num_classes=num_classes, num_support=num_support, num_query=2
        )
        log_probs = protonet(test_support_imgs, test_support_lbls, test_query_imgs)
        probabilities = torch.exp(log_probs)
        predictions = torch.argmax(probabilities, dim=1)
        
    print("\n--> FEW-SHOT INFERENCE TEST RESULTS:")
    correct_predictions = 0
    for idx in range(test_query_imgs.size(0)):
        pred_class = predictions[idx].item()
        true_class = test_query_lbls[idx].item()
        prob_val = probabilities[idx][pred_class].item()
        is_correct = "CORRECT" if pred_class == true_class else "WRONG"
        if pred_class == true_class:
            correct_predictions += 1
        print(f"   Query {idx+1:02d}: Predicted Grade = {pred_class} ({prob_val*100:.1f}%) | True Grade = {true_class} | [{is_correct}]")
        
    test_acc = (correct_predictions / test_query_imgs.size(0)) * 100
    print(f"\n--> Test Accuracy: {test_acc:.2f}%")
    print("--> Simulation completed. Real-data episodic evaluation is still required.")


if __name__ == "__main__":
    run_few_shot_training_demo()
