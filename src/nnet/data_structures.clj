(ns nnet.data-structures)

(defrecord NeuralNet [hidden-weights output-weights])

(defrecord HiddenLayer [input-values induced-local-field hidden-layer-values])
(defrecord OutputLayer [hidden-layer induced-local-field output-layer-values])
(defrecord ForwardPassResults [hidden-layer output-layer])

(defrecord TrainingExample [input-vector desired-response])
