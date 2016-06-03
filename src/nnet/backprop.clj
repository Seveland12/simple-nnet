(ns nnet.backprop
  (:require [incanter [core :refer :all]]
            [clojure.data.json :as json])
  (:use ;; [nnet.nnet :as n :only [activation-function
        ;;                         hidden-layer
        ;;                         output-layer
        ;;                         evaluate-network]]
   [nnet.math-utilities :as utils :only [approx-equals?
                                         n-ones-and-a-zero]])) 

(def my-wh (matrix [[0.362985 0.418378 0.0]
                    [-0.464489 -0.554121 0.0]
                    [-0.720958 0.504430 1.0]]))
(def my-wo (matrix [0.620124 -0.446396 0.692502]))

(def input-vector
  ; input-vector has an extra final component = 1.0 to accomodate the
  ; bias terms
  (trans (matrix [0.5 -0.5 1.0])))

(def desired-response
  (matrix [0.5]))

(defrecord TrainingExample [input-vector desired-response])

(def training-set-xor [(->TrainingExample [-0.5 -0.5 1.0] [-0.5])
                       (->TrainingExample [-0.5 0.5 1.0] [0.5])
                       (->TrainingExample [0.5 -0.5 1.0] [0.5])
                       (->TrainingExample [0.5 0.5 1.0] [-0.5])])

(def learning-rate 0.001)

(defn activation-function
  ;This is the sigmoid activation function used by each individual neuron.
  ;This version scales the tanh function to saturate at yyyy and have its 
  ;maximal derivative at +- xxxx as suggested in Haykin.
  [x]
  (* 1.7159 (Math/tanh (* 0.6666 x))))

(defn activation-function-deriv
  ; Clearly this is the derivative of the activation function.
  ; Hard-coded for now.
  [x]
  (/ 0.1439333 (utils/my-sq (Math/cosh (* 0.66666 x)))))

(defn error-function
  ; simple sum-of-squared-errors error function
  [err-vector]
  (reduce + (map utils/my-sq err-vector)))

(defrecord NeuralNet [hidden-weights output-weights])

(defrecord HiddenLayer [input-values induced-local-field hidden-layer-values])
(defrecord OutputLayer [hidden-layer induced-local-field output-layer-values])
(defrecord ForwardPassResults [hidden-layer output-layer])

(defrecord BackwardPassOL [forward-pass-results error-vector del-output delta-W-output])
(defrecord BackwardPassHL [backward-pass-ol del-hidden delta-W-hidden])
(defrecord BackwardPassResults [hidden-layer output-layer])

(def test-net (->NeuralNet my-wh my-wo))

(defn identity-matrix-with-one-zero
  [n]
  (diag (utils/n-ones-and-a-zero n)))

(defn number-of-input-neurons
  [net]
  (- (nrow (.hidden-weights net)) 1))

(defn number-of-hidden-neurons
  [net]
  (- (ncol (.hidden-weights net)) 1))

(defn number-of-output-neurons
  [net]
  (ncol (.output-weights net)))

(defn forward-pass-hidden
  [net input-vector]
  (let [ilf (mmult input-vector (.hidden-weights net))
        hlv (matrix (mapv activation-function ilf))]
    (->HiddenLayer input-vector ilf hlv)))

(defn forward-pass-output
  [net hl]
  (let [ilf (mmult (trans (.hidden-layer-values hl)) (.output-weights net))
        olv (matrix (mapv activation-function ilf))]
    (->OutputLayer hl ilf olv)))

(defn forward-pass
  [net input-vector]
  (let [hl (forward-pass-hidden net input-vector)
        ol (forward-pass-output net hl)]
    (->ForwardPassResults hl ol)))

(defn backward-pass-output
  [net desired-response fpr]
  (let [current-error-vector (minus desired-response (.output-layer-values  (.output-layer fpr)))
        current-del-output (mult current-error-vector (mapv activation-function-deriv (.induced-local-field (.output-layer fpr))))
        delta-W (mult learning-rate (mmult (.hidden-layer-values (.hidden-layer fpr)) (trans current-del-output)))
        ]
    (->BackwardPassOL fpr current-error-vector current-del-output delta-W)))

(defn calculate-del-h
  [net bpo]
  (let [n (number-of-hidden-neurons net)
        A (diag (utils/n-ones-and-a-zero n))
        temp (mmult A (.output-weights net))
        D (mult temp (.del-output bpo))
        vhidden (.induced-local-field (.hidden-layer (.forward-pass-results bpo)))
        T (matrix (mapv activation-function-deriv vhidden))]
    (mult T D)))

(defn backward-pass-hidden
  [net bpo]
  (let [del-h (calculate-del-h net bpo)
        delta-W (mult learning-rate (mmult (trans (.input-values (.hidden-layer (.forward-pass-results bpo)))) (trans del-h)))]
    (->BackwardPassHL bpo del-h delta-W)))

(defn backward-pass
  [net desired-response fpr]
  (let [bpo (backward-pass-output net desired-response fpr)
        bph (backward-pass-hidden net bpo)]
    (->BackwardPassResults bph bpo)))

(defn adjust-weights
  [net delta-W-hidden delta-W-output]
  (let [new-wh (plus (.hidden-weights net) delta-W-hidden)
        new-wo (plus (.output-weights net) delta-W-output)]
    (->NeuralNet new-wh new-wo)))

(defn iteration
  ([net training-example]
   (iteration net (trans (matrix (.input-vector training-example))) (.desired-response training-example)))

  ([net i d]
   (let [fp (forward-pass net i)
         bp (backward-pass net d fp)
         delta-wh (.delta-W-hidden (.hidden-layer bp))
         delta-wo (.delta-W-output (.output-layer bp))
         current-error-vector (.error-vector (.output-layer bp))]
     (do
       (print (error-function current-error-vector))
       (adjust-weights net delta-wh delta-wo)))))











