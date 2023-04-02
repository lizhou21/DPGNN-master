
python train.py --save_subpath citeseer --dataset citeseer --model_name DPGNN --lr 0.01 --weight_decay 0.0005 --feature_normal --topk 12 --topo_order 2 --fea_order 3 --sample 5 --tem 0.3 --lam 0.7 --hidden 32 --input_droprate 0.0 --hidden_droprate 0.2 --adj_droprate 0.5 --aggregator att --patience 200 --cuda_device 0
python train.py --save_subpath uai --dataset uai --model_name DPGNN --lr 0.01 --weight_decay 0.0005 --topk 14 --topo_order 3 --fea_order 6 --sample 2 --tem 1.0 --lam 1.0 --hidden 256 --input_droprate 0.9 --hidden_droprate 0.1 --adj_droprate 0.2 --aggregator att --patience 200 --cuda_device 0
python train.py --save_subpath acm --dataset acm --model_name DPGNN --lr 0.01 --weight_decay 0.0005 --topk 14 --topo_order 2 --fea_order 4 --sample 5 --tem 0.4 --lam 0.7 --hidden 128 --input_droprate 0.3 --hidden_droprate 0.1 --adj_droprate 0.4 --aggregator att --patience 200 --cuda_device 0
python train.py --save_subpath BlogCatalog --dataset BlogCatalog --model_name DPGNN --lr 0.01 --weight_decay 0.0005 --topk 6 --topo_order 3 --fea_order 2 --sample 6 --tem 0.1 --lam 0.2 --hidden 128 --input_droprate 0.8 --hidden_droprate 0.9 --adj_droprate 0.4 --aggregator mean --patience 200 --cuda_device 0
python train.py --save_subpath flickr --dataset flickr --model_name DPGNN --lr 0.01 --weight_decay 0.0005 --topk 8 --topo_order 2 --fea_order 6 --sample 4 --tem 1.0 --lam 0.2 --hidden 256 --input_droprate 0.9 --hidden_droprate 0.7 --adj_droprate 0.9 --aggregator att --patience 200 --cuda_device 0
