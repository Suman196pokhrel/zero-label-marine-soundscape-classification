all: wav slice feat_deep feat_ssl feat_idx cluster autolabel freeze

wav:         ; python src/00_dat_to_wav.py --cfg cfg/default.yaml
slice:       ; python src/01_slice.py --cfg cfg/default.yaml
feat_deep:   ; python src/02_features_deep.py --cfg cfg/default.yaml
feat_ssl:    ; python src/03_features_ssl.py --cfg cfg/default.yaml
feat_idx:    ; python src/04_features_indices.py --cfg cfg/default.yaml
cluster:     ; python src/05_reduce_cluster.py --cfg cfg/default.yaml && \
              python src/06_constrained_clustering.py --cfg cfg/default.yaml
autolabel:   ; python src/07_autolabel_train.py --cfg cfg/default.yaml
freeze:      ; python src/99_freeze_drop.py --cfg cfg/default.yaml
