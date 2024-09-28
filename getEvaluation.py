import argparse
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default='/data/muxy/Evaluation_indicators/gt', help='groundtrue directory')
parser.add_argument('--generated_dir', type=str, default='/data/muxy/Evaluation_indicators/generated_images', help='generated images directory')
parser.add_argument('--gpus', type=bool, default=False, help='gpu id')

args = parser.parse_args()

imagesList = os.listdir(args.gt_dir)

SSIM = 0
PSNR = 0
lpips = 0
count = 0

for img in imagesList:
    gtimage = os.path.join(args.gt_dir, img)
    genimage = os.path.join(args.generated_dir, img)
    SSIM += utils.calc_ssim(gtimage, genimage)
    PSNR += utils.calc_psnr(gtimage, genimage)
    LpipsClass = utils.util_of_lpips(net='alex', use_gpu=args.gpus)
    lpips += LpipsClass.calc_lpips(gtimage, genimage)
    count += 1

print('SSIM:', SSIM/count)
print('PSNR:', PSNR/count)
print('LPIPS:', lpips/count)

os.system('python -m pytorch_fid ' +args.gt_dir +' '+ args.generated_dir)



