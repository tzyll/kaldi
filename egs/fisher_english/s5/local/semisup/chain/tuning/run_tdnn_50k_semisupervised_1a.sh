#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script is semi-supervised recipe with around 50 hours of supervised data
# and 250 hours unsupervised data with naive splitting.
# Based on "Semi-Supervised Training of Acoustic Models using Lattice-Free MMI",
# Vimal Manohar, Hossein Hadian, Daniel Povey, Sanjeev Khudanpur, ICASSP 2018
# http://www.danielpovey.com/files/2018_icassp_semisupervised_mmi.pdf
# local/semisup/run_50k.sh shows how to call this.

# We use the combined data for i-vector extractor training.
# We use 4-gram LM trained on 1250 hours of data excluding the 250 hours
# unsupervised data to create LM for decoding. Rescoring is done with
# a larger 4-gram LM.
# This differs from the case in run_tdnn_100k_semisupervised.sh.

# This script uses phone LM to model UNK.
# This script uses the same tree as that for the seed model.
# See the comments in the script about how to change these.

# Unsupervised set: train_unsup100k_250k (250 hour subset of Fisher excluding 100 hours for supervised)
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervised): 3,2
# LM for decoding unsupervised data: 4gram
# Supervision: Naive split lattices

# Supervised training results         train_sup15k        train_sup50k
# WER on dev                          27.75               21.41
# WER on test                         27.24               21.03
# Final output train prob             -0.0959             -0.1035
# Final output valid prob             -0.1823             -0.1667
# Final output train prob (xent)      -1.9246             -1.5926
# Final output valid prob (xent)      -2.1873             -1.7990

# output-0 and output-1 are for superivsed and unsupervised data respectively.

# Semi-supervised training            train_sup15k        train_sup50k
# WER on dev                          21.31               18.98
# WER on test                         21.00               18.85
# Final output-0 train prob           -0.1577             -0.1381
# Final output-0 valid prob           -0.1761             -0.1723
# Final output-0 train prob (xent)    -1.4744             -1.3676
# Final output-0 valid prob (xent)    -1.5293             -1.4589
# Final output-1 train prob           -0.7305             -0.7671
# Final output-1 valid prob           -0.7319             -0.7714
# Final output-1 train prob (xent)    -1.1681             -1.1480
# Final output-1 valid prob (xent)    -1.2871             -1.2382

set -u -e -o pipefail

stage=0   # Start from -1 for supervised seed system training
train_stage=-100
nj=40
test_nj=40

# The following 3 options decide the output directory for semi-supervised 
# chain system
# dir=${exp_root}/chain${chain_affix}/tdnn${tdnn_affix}

exp_root=exp/semisup_50k
chain_affix=_semi50k_100k_250k    # affix for chain dir
                                  # 50 hour subset out of 100 hours of supervised data
                                  # 250 hour subset out of (1500-100=1400) hours of unsupervised data 
tdnn_affix=_semisup_1a

# Datasets -- Expects data/$supervised_set and data/$unsupervised_set to be
# present
supervised_set=train_sup50k
unsupervised_set=train_unsup100k_250k
test_sets=
test_online_decoding=true


# Input seed system
sup_chain_dir=exp/semisup_50k/chain_semi50k_100k_250k/tdnn_1a_sp  # supervised chain system
sup_lat_dir=exp/semisup_50k/chain_semi50k_100k_250k/tri4a_train_sup50k_unk_lats  # lattices for supervised set
sup_tree_dir=exp/semisup_50k/chain_semi50k_100k_250k/tree_bi_a  # tree directory for supervised chain system
graph_dir=

# Semi-supervised options
supervision_weights=1.0,1.0   # Weights for supervised, unsupervised data egs.
                              # Can be used to scale down the effect of unsupervised data
                              # by using a smaller scale for it e.g. 1.0,0.3
lm_weights=3,2  # Weights on phone counts from supervised, unsupervised data for denominator FST creation

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation

# Neural network opts
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

decode_iter=  # Iteration to decode with

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

dir=$exp_root/chain${chain_affix}/tdnn${tdnn_affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

for f in data/fbank/${supervised_set}/feats.scp \
  data/fbank/${unsupervised_set}/feats.scp \
  $sup_lat_dir/lat.1.gz $sup_tree_dir/ali.1.gz; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

# Decode unsupervised data and write lattices in non-compact
# undeterminized format
# Set --skip-scoring to false in order to score the unsupervised data
if [ $stage -le 4 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $sup_chain_dir"
  steps/nnet3/decode_semisup.sh --num-threads 1 --nj $nj --cmd "$decode_cmd" \
    --acwt 1.0 --post-decode-acwt 10.0 --write-compact false --skip-scoring true \
    --scoring-opts "--min-lmwt 10 --max-lmwt 10" --word-determinize false \
    $graph_dir data/fbank/${unsupervised_set} $sup_chain_dir/decode_${unsupervised_set}
fi

# Rescore undeterminized lattices with larger LM
if [ $stage -le 5 ]; then
  steps/lmrescore_const_arpa_undeterminized.sh --cmd "$decode_cmd" \
    --acwt 0.1 --beam 8.0  --skip-scoring true \
    data/graph/lang data/graph/lang_full \
    data/fbank/${unsupervised_set} \
    $sup_chain_dir/decode_${unsupervised_set} \
    $sup_chain_dir/decode_${unsupervised_set}_big
  ln -sf ../final.mdl $sup_chain_dir/decode_${unsupervised_set}_big/final.mdl
fi

# Get best path alignment and lattice posterior of best path alignment to be
# used as frame-weights in lattice-based training
if [ $stage -le 8 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    data/fbank/${unsupervised_set} \
    $sup_chain_dir/decode_${unsupervised_set}_big \
    $sup_chain_dir/best_path_${unsupervised_set}_big
fi

frame_subsampling_factor=1
if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sup_chain_dir/frame_subsampling_factor)
fi
cmvn_opts=$(cat $sup_chain_dir/cmvn_opts) || exit 1

diff $sup_tree_dir/tree $sup_chain_dir/tree || { echo "$0: $sup_tree_dir/tree and $sup_chain_dir/tree differ"; exit 1; }

# Uncomment the following lines if you need to build new tree using both
# supervised and unsupervised data. This may help if amount of
# supervised data used to train the seed system tree is very small.
# unsupervised data

# tree_affix=bi_semisup_a
# treedir=$exp_root/chain${chain_affix}/tree_${tree_affix}
# if [ -f $treedir/final.mdl ]; then
#   echo "$0: $treedir/final.mdl exists. Remove it and run again."
#   exit 1
# fi
#
# if [ $stage -le 9 ]; then
#   # This is usually 3 for chain systems.
#   echo $frame_subsampling_factor > \
#     $sup_chain_dir/best_path_${unsupervised_set}_big/frame_subsampling_factor
#
#   # This should be 1 if using a different source for supervised data alignments.
#   # However alignments in seed tree directory have already been sub-sampled.
#   echo $frame_subsampling_factor > \
#     $sup_tree_dir/frame_subsampling_factor
#
#   # Build a new tree using stats from both supervised and unsupervised data
#   steps/nnet3/chain/build_tree_multiple_sources.sh \
#     --use-fmllr false --context-opts "--context-width=2 --central-position=1" \
#     --frame-subsampling-factor $frame_subsampling_factor \
#     7000 data/lang_test_tgsmall \
#     data/fbank/${supervised_set} \
#     ${sup_tree_dir} \
#     data/fbank/${unsupervised_set} \
#     ${sup_chain_dir}/best_path_${unsupervised_set}_big \
#     $treedir || exit 1
# fi
#
# sup_tree_dir=$treedir   # Use the new tree dir for further steps

# Train denominator FST using phone alignments from
# supervised and unsupervised data
if [ $stage -le 10 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${sup_tree_dir} ${sup_chain_dir}/best_path_${unsupervised_set}_big \
    $dir
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $sup_tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  #input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts

  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.

  output name=output-0 input=output.affine
  output name=output-1 input=output.affine

  output name=output-0-xent input=output-xent.log-softmax
  output name=output-1-xent input=output-xent.log-softmax
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# Get values for $model_left_context, $model_right_context
. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context

egs_left_context=$(perl -e "print int($left_context + $frame_subsampling_factor / 2)")
egs_right_context=$(perl -e "print int($right_context + $frame_subsampling_factor / 2)")
frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
touch $sup_egs_dir/.nodelete

unsup_frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4.0  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
tolerance=5   # frame-tolerance for chain training

unsup_lat_dir=${sup_chain_dir}/decode_${unsupervised_set}_big
if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsupervised_set}

  if [ $stage -le 13 ]; then
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    steps/nnet3/chain/get_egs.sh \
      --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --constrained false \
      --left-tolerance $tolerance --right-tolerance $tolerance \
      --left-context $egs_left_context --right-context $egs_right_context \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 2500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam "$lattice_prune_beam" \
      --deriv-weights-scp $sup_chain_dir/best_path_${unsupervised_set}_big/weights.scp \
      --generate-egs-scp true $unsup_egs_opts \
      data/fbank/${unsupervised_set} $dir \
      $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/comb_egs
if [ $stage -le 14 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 64 \
    --lang2weight $supervision_weights 2 \
    $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  # This is to skip stages of den-fst creation, which was already done.
  train_stage=-4
fi

if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$comb_egs_dir" \
    --cmd "$decode_cmd" \
    --use-gpu wait \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.000015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir data/fbank/${supervised_set} \
    --tree-dir $sup_tree_dir \
    --lat-dir $sup_lat_dir \
    --dir $dir || exit 1;
fi

iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in $test_sets; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/fbank/${decode_set} $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/graph/lang data/graph/lang_full \
          data/fbank/${decode_set} $dir/decode_${decode_set}${decode_iter:+_$decode_iter} \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_full || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 18 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --feature-type fbank \
       --fbank-config conf/fbank.conf \
       data/graph/lang $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for data in $test_sets; do
    (
      #nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          $graph_dir data/fbank/${data} ${dir}_online/decode_${data} || exit 1

    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit 0;
