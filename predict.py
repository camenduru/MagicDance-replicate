import os
from cog import BasePredictor, Input, Path
import sys, subprocess, cv2
sys.path.append('/content/MagicDance')
os.chdir('/content/MagicDance')

def images_to_video(input_folder, output_file, fps=30):
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))
    video.release()

def run_magic_dance(image_path, num_train_steps, seed, pose_sequence):
    print(str(image_path), str(num_train_steps), str(seed), str(pose_sequence))
    command = ['CUDA_VISIBLE_DEVICES=0', 'torchrun', '--master_port', '18102', 'test_any_image_pose.py',
               '--model_config', '/content/MagicDance/model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml',
               '--num_train_steps', str(num_train_steps),
               '--seed', str(seed),
               '--img_bin_limit', 'all',
               '--train_batch_size', '1',
               '--use_fp16',
               '--control_mode', 'controlnet_important',
               '--control_type', 'body+hand+face',
               '--train_dataset', 'tiktok_video_arnold',
               '--v4',
               '--with_text',
               '--wonoise',
               '--image_pretrain_dir', '/content/MagicDance/pretrained_weights/model_state-110000.th',
               '--init_path', '/content/MagicDance/pretrained_weights/control_sd15_ini.ckpt',
               '--local_image_dir', '/content/MagicDance/tiktok_test_log/image_log/0125/001/image',
               '--local_log_dir', '/content/MagicDance/tiktok_test_log/tb_log/0125/001/log',
               '--local_pose_path', f'/content/MagicDance/example_data/pose_sequence/{str(pose_sequence)}',
               '--local_cond_image_path', str(image_path)]
    result = subprocess.run(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(command, result.stdout)
    input_folder = f'/content/MagicDance/tiktok_test_log/image_log/0125/001/image/{str(num_train_steps-1)}/gen_images'
    output_file = f'/content/MagicDance/tiktok_test_log/image_log/0125/001/image/{str(num_train_steps-1)}/gen_images/video.mp4'
    images_to_video(input_folder, output_file)
    return output_file

class Predictor(BasePredictor):
    def setup(self) -> None:
        directory = "/content"
        if not os.path.exists(directory):
            os.mkdir(directory)
    def predict(
        self,
        image_path: Path = Input(description="Input Image"),
        num_train_steps: int = Input(default=1, ge=1, le=5),
        seed: int = Input(default=42),
        pose_sequence: str = Input(choices=["001","002","003"], default="001"),
    ) -> Path:
        output_video = run_magic_dance(image_path, num_train_steps, seed, pose_sequence)
        return Path(output_video)