#!/bin/bash
# Copyright 2019  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

# This script for oriental language recognition is based on ../../sre16/v2/run.sh which is used for speaker recognition.


. ./cmd.sh
. ./path.sh

stage=1

set -eu


###### Bookmark: basic preparation ######

# Prepare training set in data/train (for ap19-olr, including almost all train/test data used in the previous challenges),
# both contain at least wav.scp, utt2lang, spk2utt and utt2spk,
# spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id.


###### Bookmark: feature and vad computation ######

if [ $stage -le 1 ]; then
  # Produce Fbank and MFCC in data/{fbank,mfcc}/train
  for x in train; do
    mkdir -p data/fbank/$x && cp -r data/$x/{spk2utt,utt2lang,utt2spk,wav.scp} data/fbank/$x
    mkdir -p data/mfcc/$x && cp -r data/$x/{spk2utt,utt2lang,utt2spk,wav.scp} data/mfcc/$x
    steps/make_fbank.sh --nj 10 --cmd "$train_cmd" --write-utt2num-frames true data/fbank/$x
    steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" data/mfcc/$x
    sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
    cp data/mfcc/$x/vad.scp data/fbank/$x/vad.scp
  done
fi


###### Bookmark: x-vector training ######

# Caution: in order to use off-the-shelf scripts in ../../sre16/v2 for speaker recogniton,
# we copy utt2lang to utt2spk, i.e., each fake spk is actually a language.
if [ $stage -le 2 ]; then
  mv data/fbank/train/utt2spk data/fbank/train/utt2spk.bak
  mv data/fbank/train/spk2utt data/fbank/train/spk2utt.bak
  cp data/fbank/train/utt2lang data/fbank/train/utt2spk
  utils/utt2spk_to_spk2utt.pl data/fbank/train/utt2spk > data/fbank/train/spk2utt
  utils/fix_data_dir.sh data/fbank/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 10 --cmd "$train_cmd" \
    data/fbank/train data/fbank/train_no_sil exp/fbank/train_no_sil
  utils/fix_data_dir.sh data/fbank/train_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames. We want at least 0.5s (50 frames) per utterance.
  min_len=50
  mv data/fbank/train_no_sil/utt2num_frames data/fbank/train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/fbank/train_no_sil/utt2num_frames.bak > data/fbank/train_no_sil/utt2num_frames
  utils/filter_scp.pl data/fbank/train_no_sil/utt2num_frames data/fbank/train_no_sil/utt2spk > data/fbank/train_no_sil/utt2spk.new
  mv data/fbank/train_no_sil/utt2spk.new data/fbank/train_no_sil/utt2spk
  utils/fix_data_dir.sh data/fbank/train_no_sil
fi

nnet_dir=exp/xvect
# stage 4-6 inside
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/fbank/train_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs


###### Bookmark: evaluation of three tasks ######

# Produce feats for test sets
if [ $stage -le 7 ]; then
  # following test sets contain at least wav.scp, utt2lang, spk2utt and utt2spk,
  # spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id.
  for x in task_1 task_2 task_3/enroll task_3/test; do
    mkdir -p data_test_final/fbank/$x && cp -r data_test_final/$x/{spk2utt,utt2lang,utt2spk,wav.scp} data_test_final/fbank/$x
    mkdir -p data_test_final/mfcc/$x && cp -r data_test_final/$x/{spk2utt,utt2lang,utt2spk,wav.scp} data_test_final/mfcc/$x
    steps/make_fbank.sh --nj 8 --cmd "$train_cmd" --write-utt2num-frames true data_test_final/fbank/$x
    steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" data_test_final/mfcc/$x
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" data_test_final/mfcc/$x data_test_final/mfcc/$x/log data_test_final/mfcc/$x/data
    cp data_test_final/mfcc/$x/vad.scp data_test_final/fbank/$x/vad.scp
  done

  # spk2utt is fake, actually from utt2lang, see stage 2
  awk -v id=0 '{print $1, id++}' data/fbank/train/spk2utt  > $nnet_dir/lang2lang_id
fi


# Task 1: Short-utterance
# Task 2: Cross-channel LID
# outputs of the original x-vect system by propagating the test set are used as scores.
if [ $stage -le 8 ]; then
  # forward the net
  for x in task_1 task_2; do
    local/run_xvect_score.sh --cmd "$train_cmd --mem 6G" --nj 10 \
      $nnet_dir data_test_final/fbank/$x \
      exp/xvectors_$x
  done

  # print eer and cavg
  for x in task_1 task_2; do
    # prepare trials
    local/prepare_trials.py data/fbank/train data_test_final/fbank/$x
    trials=data_test_final/fbank/$x/trials
    # only keep the 6 target languages for task 2
    if [[ $x == "task_2" ]]; then
      grep -E 'Tibet |Uyghu |ja-jp |ru-ru |vi-vn |zh-cn ' data_test_final/fbank/$x/trials > data_test_final/fbank/$x/trials.6
      mv data_test_final/fbank/$x/trials.6 data_test_final/fbank/$x/trials
      langs='Tibet Uyghu ja-jp ru-ru vi-vn zh-cn'
      python local/filter_lre_matrix.py "$langs" exp/xvectors_$x/output.ark.utt > exp/xvectors_$x/output.ark.utt.6
      mv exp/xvectors_$x/output.ark.utt.6 exp/xvectors_$x/output.ark.utt
    fi
    echo "---- $x ----"
    eer=`compute-eer <(python local/nnet3/prepare_for_eer.py $trials exp/xvectors_$x/output.ark.utt) 2> /dev/null`
    printf "%15s %5.2f \n" "$x utt level eer%:" $eer
    cavg=`python local/compute_cavg.py -matrix $trials exp/xvectors_$x/output.ark.utt`
    printf "%15s %7.4f \n" "$x utt level cavg:" $cavg
  done
fi


# Task 3: Zero-resource LID
# x-vects are extracted for both enroll and test sets, then used for identification.
# the training set for the x-vect system doesn't include the languages in task 3.
if [ $stage -le 9 ]; then
  for x in task_3/enroll task_3/test; do
    local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 10 \
      $nnet_dir data_test_final/fbank/$x \
      exp/xvectors_$x
  done

  for x in task_3/test; do
    # prepare trials
    local/prepare_trials.py data_test_final/fbank/task_3/enroll data_test_final/fbank/$x
    trials=data_test_final/fbank/$x/trials
  done

  exp=exp/xvectors_task_3
  # basic cosine scoring on x-vectors
  local/cosine_scoring.sh data_test_final/fbank/task_3/enroll data_test_final/fbank/task_3/test \
    $exp/enroll $exp/test $trials $exp/scores

  # print eer and cavg
  for i in cosine; do
    echo "---- task_3 ----"
    eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
    printf "%15s %5.2f \n" "$i eer%:" $eer
    cavg=`python local/compute_cavg.py -pairs $trials $exp/scores/${i}_scores`
    printf "%15s %7.4f \n" "$i cavg:" $cavg
  done
fi

exit 0
