(ns nnet.backprop
  (:require [clojure.core.matrix :as m]
            [clojure.data.json :as json]
            [nnet.data-structures :refer :all]
            [nnet.nnet :refer :all])
  (:use [nnet.math-utilities :as utils :only [approx-equals?
                                              n-ones-and-a-zero]]))

;; these three are just XOR net definitions to ease debugging of various functions
(def my-wh (m/matrix [[0.362985 0.418378 0.0]
                    [-0.464489 -0.554121 0.0]
                    [-0.720958 0.504430 1.0]]))
(def my-wo (m/matrix [0.620124 -0.446396 0.692502]))
(def test-net (->NeuralNet my-wh my-wo))

(def learning-rate 0.1)

(defn error-function
  ; simple sum-of-squared-errors error function
  [err-vector]
  (reduce + (map utils/my-sq err-vector)))

(defn initial-weights [from to]
  ; returns a matrix of random initial weights
  (let [x (* from to)]
    (mapv vec (partition from (repeatedly x #(rand (/ 1 x)))))))

(defn initial-net
  ; n = # input layer neurons + 1
  ; m = # hidden layer neurons + 1
  ; p = # output layer neurons
  [training-set]
  (let [n (m/ecount (.input-vector (nth training-set 0)))
        m n
        p (m/ecount (.desired-response (nth training-set 0)))
        wh_0 (m/identity-matrix n)
        wo_0 (m/matrix (initial-weights m p))] 
    (->NeuralNet wh_0 wo_0)))

(defrecord BackwardPassOL [forward-pass-results error-vector del-output delta-W-output])
(defrecord BackwardPassHL [backward-pass-ol del-hidden delta-W-hidden])
(defrecord BackwardPassResults [hidden-layer output-layer])

(defrecord IterationResults [current-net error-value])

(defn identity-matrix-with-one-zero
  ; returns a matrix that is an identity matrix except for the bottom-right
  ; element, which is 0.
  [n]
  (m/diagonal-matrix (utils/n-ones-and-a-zero n)))

(defn backward-pass-output
  [net desired-response fpr]
  (let [current-error-vector (m/sub desired-response (.output-layer-values (.output-layer fpr)))
        current-del-output (m/mul current-error-vector (mapv activation-function-deriv (.induced-local-field (.output-layer fpr))))
        delta-W (m/mul learning-rate (m/mmul (.hidden-layer-values (.hidden-layer fpr)) (m/transpose current-del-output)))]
    (->BackwardPassOL fpr current-error-vector current-del-output delta-W)))

(defn calculate-del-h
  [net bpo]
  (let [n (number-of-hidden-neurons net)
        A (m/diagonal-matrix (utils/n-ones-and-a-zero n))
        temp (m/mmul A (.output-weights net))
        D (m/mmul temp (.del-output bpo)) ;this is the problem
        vhidden (.induced-local-field (.hidden-layer (.forward-pass-results bpo)))
        T (m/matrix (mapv activation-function-deriv vhidden))]
    (m/mul T D)))

(defn backward-pass-hidden
  [net bpo]
  (let [del-h (calculate-del-h net bpo)
        delta-W (m/mul learning-rate (m/mmul (m/transpose (.input-values (.hidden-layer (.forward-pass-results bpo)))) (m/transpose del-h)))]
    (->BackwardPassHL bpo del-h delta-W)))

(defn backward-pass
  [net desired-response fpr]
  (let [bpo (backward-pass-output net desired-response fpr)
        bph (backward-pass-hidden net bpo)]
    (->BackwardPassResults bph bpo)))

(defn adjust-weights
  [net delta-W-hidden delta-W-output]
  (let [new-wh (m/add (.hidden-weights net) delta-W-hidden)
        new-wo (m/add (.output-weights net) delta-W-output)]
    (->NeuralNet new-wh new-wo)))

(defn add-network-weights
  [net1 net2]
  (let [wh-new (m/add (.hidden-weights net1) (.hidden-weights net2))
        wo-new (m/add (.output-weights net1) (.output-weights net2))]
    (->NeuralNet wh-new wo-new)))

(defn epoch-reducer
  [result1 result2]
  (let [net1 (.current-net result1)
        net2 (.current-net result2)
        error1 (.error-value result1)
        error2 (.error-value result2)
        wh-new (m/add (.hidden-weights net1) (.hidden-weights net2))
        wo-new (m/add (.output-weights net1) (.output-weights net2))
        new-net (->NeuralNet wh-new wo-new)
        total-error (+ error1 error2)]
    (->IterationResults new-net total-error)))

(defn iteration
  ([net training-example]
   (iteration net (m/transpose (m/matrix (.input-vector training-example))) (m/matrix (.desired-response training-example))))

  ([net i d]
   (let [fp (forward-pass net i)
         bp (backward-pass net d fp)
         delta-wh (.delta-W-hidden (.hidden-layer bp))
         delta-wo (.delta-W-output (.output-layer bp))
         current-error-vector (.error-vector (.output-layer bp))
         current-error-value (error-function current-error-vector)]
     (do
       (->IterationResults (->NeuralNet delta-wh delta-wo) current-error-value)))))

(defn train
  ([training-set]
   (train (initial-net training-set) training-set))
  
  ([net training-set]
   (let [num-examples (m/ecount training-set)]
     (loop [current-net net current-err 999.0]
       (let [current-iteration-adjustment (reduce epoch-reducer (map (partial iteration current-net) training-set))
             current-iteration-result (add-network-weights current-net (.current-net current-iteration-adjustment))]
         (let [current-avg-err (/ (.error-value current-iteration-adjustment) num-examples)]
           (do
             (println current-avg-err)
             (if-not (approx-equals? current-avg-err 0.0)
               (recur current-iteration-result current-avg-err)
               current-net))))))))
