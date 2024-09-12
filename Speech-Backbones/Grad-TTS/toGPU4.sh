# input_path=new_exp_sg_acc_blank_conformer_gst_E7/gen_grad_334/raw
# output_path=gen_grad_334_E7/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

## E8 file
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln.py \
#     -f resources/filelists/synthesis_zh_acc.txt \
#     -c /data2/xintong/tts_server/Grad-TTS/new_exp_sg_acc_blank_conformer_gst_E8/grad_300.pt \
#     -o /data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_300_E8/raw


# Check if the -r parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 -r <phonemes_string>"
  exit 1
fi
# text
python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_text.py \
    -r "$1" \
    -c /data2/xintong/tts_server/Grad-TTS/new_exp_sg_acc_blank_conformer_gst_E8/grad_300.pt \
    -o /data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_300_E8_test/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E8/gen_grad_400/raw 
# output_path=gen_grad_400_E8/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# E10
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_grl.py -f resources/filelists/synthesis_zh_acc.txt -c logs/new_exp_sg_acc_blank_conformer_gst_E10/grad_300.pt -o logs/new_exp_sg_acc_blank_conformer_gst_E10/gen_grad_300/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E10/gen_grad_300/raw 
# output_path=gen_grad_300_E10/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# E12
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_grl_E12.py \
#     -f resources/filelists/synthesis_zh_acc.txt \
#     -c logs/new_exp_sg_acc_blank_conformer_gst_E12/grad_100.pt \
#     -o logs/new_exp_sg_acc_blank_conformer_gst_E12/gen_grad_100_male/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E12/gen_grad_100_male/raw 
# output_path=gen_grad_100_E12_male/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# E14 base
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_grl_E12.py \
#     -f resources/filelists/synthesis_zh_acc.txt \
#     -c logs/new_exp_sg_acc_blank_conformer_gst_E14/grad_253.pt \
#     -o logs/new_exp_sg_acc_blank_conformer_gst_E14/gen_grad_253_male/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E14/gen_grad_253_male/raw 
# output_path=gen_grad_253_E14_male/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# E13
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_grl_E12.py \
#     -f resources/filelists/synthesis_zh_acc.txt \
#     -c logs/new_exp_sg_acc_blank_conformer_gst_E13/grad_251.pt \
#     -o logs/new_exp_sg_acc_blank_conformer_gst_E13/gen_grad_251_male/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E13/gen_grad_251_male/raw 
# output_path=gen_grad_251_E13_male/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# E15
# python dump_feats_to_GPU4_zh_acc_blank_conformer_gstloss_cln_grl_E12.py \
#     -f resources/filelists/synthesis_zh_acc.txt \
#     -c logs/new_exp_sg_acc_blank_conformer_gst_E15/grad_500.pt \
#     -o logs/new_exp_sg_acc_blank_conformer_gst_E15/gen_grad_500_male/raw

# input_path=new_exp_sg_acc_blank_conformer_gst_E15/gen_grad_500_male/raw 
# output_path=gen_grad_500_E15_male/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r


# input_path=new_exp_sg_acc_blank_conformer_gst_E5/gen_grad_407/raw 
# output_path=gen_grad_407_E6/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

# input_path=new_exp_sg_acc_blank_conformer_gst_E5/gen_grad_333/raw 
# output_path=gen_grad_333_E6/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r



