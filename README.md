# pipeline-RGB
采集 
→ 标定 → 生成 stereo_calib_rgb.npz
生成新内参
→ 运行时：采集 + rectified
→ 输入 FS 

cd /home/match/foundationstereo-industrial-6dpose/RGB-pipeline \
conda activate foundationpose 

只做一次：
固定两台相机 → 水平左右摆好，不再移动

python save_calib_images_dual_rgb.py 拍 15–25 组棋盘格

python stereo_calibrate_rgb.py → 得到 stereo_calib_rgb.npz

python make_fs_intrinsic_from_npz.py → 得到 assets/K_d435.txt

每次想测试 FS 时：

cd /home/match/foundationstereo-industrial-6dpose/RGB-pipeline \
conda activate foundationpose 

python capture_rectified_for_fs.py              → shared_fs_test/left_rect.png / right_rect.png

scripts/run_demo.py + K_d435.txt → 观察深度/视差输出是否合理

cd ~/FS/FoundationStereo \
conda activate foundation_stereo 

python scripts/run_demo.py \
  --left_file ./shared_fs_test/left_rect.png \
  --right_file ./shared_fs_test/right_rect.png \
  --ckpt_dir ./pretrained_models/model_best_bp2.pth \
  --out_dir ./outputs_test \
  --intrinsic ./assets/K_d435_2RGB.txt

