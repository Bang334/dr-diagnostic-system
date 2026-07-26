import os

class RetinalConfig:
    """
    Cấu hình tổng thể cho hệ thống huấn luyện và suy luận phân đoạn tổn thương võng mạc.
    Hỗ trợ chuyển đổi cấu hình linh hoạt cho 3 loại tổn thương: MA, HE, EX.
    """
    def __init__(self, lesion_type: str = 'MA', dataset_name: str = 'idrid'):
        assert lesion_type in ['MA', 'HE', 'EX'], "lesion_type phải thuộc ['MA', 'HE', 'EX']"
        assert dataset_name in ['idrid', 'ddr'], "dataset_name phải thuộc ['idrid', 'ddr']"
        
        self.lesion_type = lesion_type
        self.dataset_name = dataset_name
        
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if dataset_name == 'idrid':
            self.data_root = os.path.join(self.project_root, 'idrid')
            self.train_img_dir = os.path.join(self.data_root, 'train', 'images')
            self.train_mask_dir = os.path.join(self.data_root, 'train', 'masks', lesion_type)
            self.test_img_dir = os.path.join(self.data_root, 'test', 'images')
            self.test_mask_dir = os.path.join(self.data_root, 'test', 'masks', lesion_type)
        else:
            self.data_root = os.path.join(self.project_root, 'ddr')
            self.train_img_dir = os.path.join(self.data_root, 'train', 'images')
            self.train_mask_dir = os.path.join(self.data_root, 'train', 'masks', lesion_type)
            self.val_img_dir = os.path.join(self.data_root, 'val', 'images')
            self.val_mask_dir = os.path.join(self.data_root, 'val', 'masks', lesion_type)
            self.test_img_dir = os.path.join(self.data_root, 'test', 'images')
            self.test_mask_dir = os.path.join(self.data_root, 'test', 'masks', lesion_type)
            
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints', f'{dataset_name}_{lesion_type}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.use_roi = True
        self.target_size = (512, 512)
        self.patch_size = 512
        self.stride = 448
        
        self.batch_size = 8
        self.max_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_workers = 2
        self.precision = 16
        
        self.copy_paste_prob = 0.5 if lesion_type in ['MA', 'HE'] else 0.0
        self.lesion_crop_size = 64 if lesion_type == 'MA' else 128
        
        self.model_architecture = 'attention_unet'
        self.backbone_name = 'resnet50' if lesion_type == 'MA' else 'resnet34'
        self.encoder_weights = 'imagenet'
        
        self.loss_type = 'combo'
        
        if lesion_type == 'MA':
            self.tversky_alpha = 0.8
            self.tversky_beta = 0.2
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
            self.combo_weight_focal = 0.4
        elif lesion_type == 'HE':
            self.tversky_alpha = 0.7
            self.tversky_beta = 0.3
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
            self.combo_weight_focal = 0.5
        else:
            self.tversky_alpha = 0.5
            self.tversky_beta = 0.5
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
            self.combo_weight_focal = 0.5
            
        self.threshold = 0.35 if lesion_type == 'MA' else (0.5 if lesion_type == 'HE' else 0.6)
        self.min_lesion_area = 5 if lesion_type == 'MA' else (15 if lesion_type == 'HE' else 10)
        self.use_optic_disc_mask = True if lesion_type == 'EX' else False
        
    def get_summary(self):
        return f"Config [{self.dataset_name.upper()} - {self.lesion_type}]: Arch={self.model_architecture}, Loss={self.loss_type}, LR={self.learning_rate}"
