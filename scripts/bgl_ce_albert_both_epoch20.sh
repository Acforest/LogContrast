python train.py \
--log_type "BGL" \
--model_dir "./models/" \
--model_name "" \
--semantic_model_name "albert" \
--feat_type "both" \
--feat_dim 512 \
--vocab_size 2000 \
--do_train \
--train_batch_size 64 \
--train_data_dir "./output/BGL/BGL_train_10000.csv" \
--loss_fct "ce" \
--num_epoch 20 \
--lr 0.00001 \
--weight_decay 0.01 \
--lambda_c 0.1 \
--temperature 0.5 \
--do_test \
--test_batch_size 64 \
--test_data_dir "./output/BGL/BGL_test_942699.csv" \
--seed 1234 \
--device "cuda"