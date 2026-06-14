from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-dir', type=str, required=True, help='Path to local YOLOv5 repo, e.g. D:/CODE/yolov5')
    parser.add_argument('--data-yaml', type=str, default='datasets/text_yolo/text_yolov5.yaml')
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='bjtu_text_yolov5')
    parser.add_argument('--device', type=str, default='', help='Use 0 for GPU, cpu for CPU, empty for auto')
    args = parser.parse_args()

    yolov5_dir = Path(args.yolov5_dir)
    train_py = yolov5_dir / 'train.py'
    if not train_py.exists():
        raise FileNotFoundError(f'Cannot find YOLOv5 train.py: {train_py}')

    data_yaml = Path(args.data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f'Cannot find data yaml: {data_yaml}. Run convert_labelme_to_yolov5.py first.')

    cmd = [
        sys.executable,
        str(train_py),
        '--img', str(args.img),
        '--batch', str(args.batch),
        '--epochs', str(args.epochs),
        '--data', str(data_yaml),
        '--weights', args.weights,
        '--project', args.project,
        '--name', args.name,
        '--exist-ok'
    ]

    if args.device:
        cmd.extend(['--device', args.device])

    print('Running command:')
    print(' '.join(cmd))
    subprocess.run(cmd, cwd=str(yolov5_dir), check=True)

    print('\nTraining finished.')
    print('Best weights usually at:')
    print(yolov5_dir / args.project / args.name / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
