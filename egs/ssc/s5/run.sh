#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0

# data/$train
train=train_sup
train_unsup=train_unsup
tests="test_waihu_add_0302"


. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e


if [ $stage -le 5 ]; then
  # data/lang already prepared
  # convert arpa.gz to G.fst
  lm=data/lang/model_for_400_shenhe_mix_0223_s_o4.arpa.gz
  lex=data/lang/lexicon.txt
  # lm too large, prune it
  prune_thresh=0.0000000001
  pruned_lm=data/lang/model_for_400_shenhe_mix_0223_s_o4.pruned.e-10.arpa.gz
  ngram -prune $prune_thresh -lm $lm -write-lm $pruned_lm
  utils/format_lm.sh data/lang $pruned_lm $lex data/graph/lang
  # Create ConstArpaLm format language model for full LM
  utils/build_const_arpa_lm.sh $lm data/lang data/graph/lang_full
fi

if [ $stage -le 6 ]; then
  for part in $train $tests; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part
    steps/compute_cmvn_stats.sh data/$part
  done
fi

if [ $stage -le 7 ]; then
  # Make some small data subsets for early system-build stages.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.
  utils/subset_data_dir.sh --shortest data/$train 2000 data/train_2kshort
  utils/subset_data_dir.sh data/$train 5000 data/train_5k
  utils/subset_data_dir.sh data/$train 10000 data/train_10k
  utils/subset_data_dir.sh data/$train 50000 data/train_50k
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang exp/mono
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang exp/tri1 exp/tri1_ali_10k

  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang exp/tri1_ali_10k exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang exp/tri2b_ali_10k exp/tri3b
fi

if [ $stage -le 12 ]; then
  # Align a 50k utts subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_50k data/lang \
    exp/tri3b exp/tri3b_ali_train_50k

  # train another LDA+MLLT+SAT system
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_50k data/lang \
                      exp/tri3b_ali_train_50k exp/tri4b
fi

if [ $stage -le 16 ]; then
  # align using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/$train data/lang exp/tri4b exp/tri4b_ali_train

  # create a larger SAT model
  steps/train_quick.sh  --cmd "$train_cmd" 5000 100000 \
                      data/$train data/lang exp/tri4b_ali_train exp/tri5b
  # decode using the tri5b model
  utils/mkgraph.sh data/graph/lang \
                   exp/tri5b exp/tri5b/graph
  for test in $tests; do
      steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
                            exp/tri5b/graph data/$test exp/tri5b/decode_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/graph/lang data/graph/lang_full \
        data/$test exp/tri5b/decode_$test exp/tri5b/decode_full_$test
  done
fi

if [ $stage -le 19 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/$train data/lang exp/tri5b exp/tri5b_ali_$train
fi

if [ $stage -le 21 ]; then
  # seed model for semisup training
  for part in $train $train_unsup $tests; do
    utils/copy_data_dir.sh --validate-opts "--non-print" data/$part data/fbank/$part
    steps/make_fbank.sh --cmd "$train_cmd" --nj 40 data/fbank/$part
    steps/compute_cmvn_stats.sh data/fbank/$part
  done
  local/chain/run_tdnn.sh --stage 0 --train-stage -10 \
    --train-set $train \
    --test-sets $tests \
    --gmm tri5b \
    --nnet3_affix _sup
fi

if [ $stage -le 22 ]; then
  # semisup reusing fisher_english's, no ivector, no speed pertub.
  exp_root=exp/chain_sup
  sup_egs_dir=$exp_root/tdnn_1d/egs
  # get cegs.scp for egs/.
  for i in $sup_egs_dir/*.cegs; do
    mv $i ${i}_tmp
    nnet3-chain-copy-egs ark:${i}_tmp ark,scp:$i,${i/cegs/scp}
  done
  for i in $sup_egs_dir/cegs.*.ark; do
    mv $i ${i}_tmp
    nnet3-chain-copy-egs ark:${i}_tmp ark,scp:$i,${i/ark/scp}
  done
  cat $sup_egs_dir/cegs.*.scp > $sup_egs_dir/cegs.scp
  rm $sup_egs_dir/{cegs.*.scp,*.tmp}
  # dir=${exp_root}/chain${chain_affix}/tdnn${tdnn_affix}
  local/semisup/chain/run_tdnn_50k_semisupervised.sh \
    --supervised-set $train \
    --unsupervised-set $train_unsup \
    --test-sets $tests \
    --exp-root $exp_root \
    --chain-affix _semi_sup_unsup \
    --sup-chain-dir $exp_root/tdnn_1d \
    --sup-egs-dir $sup_egs_dir \
    --sup-lat-dir $exp_root/tri5b_${train}_lats \
    --sup-tree-dir $exp_root/tree \
    --graph-dir  exp/chain_sup/tdnn_1d/graph
fi

# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh # set "--stage 11" if you have already run local/chain/run_tdnn.sh

# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh

# # The following is the current online-nnet2 recipe, with "multi-splice".
# local/online/run_nnet2_ms.sh

# # The following is the discriminative-training continuation of the above.
# local/online/run_nnet2_ms_disc.sh

# ## The following is an older version of the online-nnet2 recipe, without "multi-splice".  It's faster
# ## to train but slightly worse.
# # local/online/run_nnet2.sh
