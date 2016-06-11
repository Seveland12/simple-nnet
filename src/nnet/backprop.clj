(ns nnet.backprop
  (:require [incanter [core :refer :all]]
            [clojure.data.json :as json]
            [nnet.data-structures :refer :all]
            [nnet.nnet :refer :all])
  (:use [nnet.math-utilities :as utils :only [approx-equals?
                                              n-ones-and-a-zero]]))

(def my-wh (matrix [[0.362985 0.418378 0.0]
                    [-0.464489 -0.554121 0.0]
                    [-0.720958 0.504430 1.0]]))
(def my-wo (matrix [0.620124 -0.446396 0.692502]))

(defrecord TrainingExample [input-vector desired-response])

(def training-set-xor [(->TrainingExample [-0.5 -0.5 1.0] [-0.5])
                       (->TrainingExample [-0.5 0.5 1.0] [0.5])
                       (->TrainingExample [0.5 -0.5 1.0] [0.5])
                       (->TrainingExample [0.5 0.5 1.0] [-0.5])])

(def training-set-calc-dig [(->TrainingExample [0.5 0.5 0.5, 0.5 -0.5 0.5, 0.5 -0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, 1.0] [0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]) ; 0
                            (->TrainingExample [-0.5 0.5 -0.5, 0.5 0.5 -0.5, -0.5 0.5 -0.5, -0.5 0.5 -0.5, 0.5 0.5 0.5, 1.0] [-0.5 0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]) ; 1
                            (->TrainingExample [0.5 0.5 0.5, -0.5 -0.5 0.5, 0.5 0.5 0.5, 0.5 -0.5 -0.5, 0.5 0.5 0.5, 1.0] [-0.5 -0.5 0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]) ; 2
                            (->TrainingExample [0.5 0.5 0.5, -0.5 -0.5 0.5, 0.5 0.5 0.5, -0.5 -0.5 0.5, 0.5 0.5 0.5, 1.0] [-0.5 -0.5 -0.5 0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]) ; 3
                            (->TrainingExample [0.5 -0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, -0.5 -0.5 0.5, -0.5 -0.5 0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5 -0.5 -0.5 -0.5]) ; 4
                            (->TrainingExample [0.5 0.5 0.5, 0.5 -0.5 -0.5, 0.5 0.5 0.5, -0.5 -0.5 0.5, 0.5 0.5 0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5 -0.5 -0.5]) ; 5
                            (->TrainingExample [0.5 0.5 0.5, 0.5 -0.5 -0.5, 0.5 0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5 -0.5]) ; 6
                            (->TrainingExample [0.5 0.5 0.5, -0.5 -0.5 0.5, -0.5 0.5 -0.5, -0.5 0.5 -0.5, -0.5 0.5 -0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5]) ; 7
                            (->TrainingExample [0.5 0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5]) ; 8
                            (->TrainingExample [0.5 0.5 0.5, 0.5 -0.5 0.5, 0.5 0.5 0.5, -0.5 -0.5 0.5, -0.5 -0.5 0.5, 1.0] [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.5]) ; 9
                            ])

(def learning-rate 0.1)

(defn error-function
  ; simple sum-of-squared-errors error function
  [err-vector]
  (reduce + (map utils/my-sq err-vector)))

(defn initial-net
  ; n = # input layer neurons
  ; m = # hidden layer neurons
  ; p = # output layer neurons
  [training-set]
  (let [n (length (.input-vector (nth training-set 0)))
        m n
        p (length (.desired-response (nth training-set 0)))
        ; + 1 because of bias terms
        wh_0 (identity-matrix n)
        wo_0 (matrix 0.1 m p)]
    (->NeuralNet wh_0 wo_0)))

(defrecord BackwardPassOL [forward-pass-results error-vector del-output delta-W-output])
(defrecord BackwardPassHL [backward-pass-ol del-hidden delta-W-hidden])
(defrecord BackwardPassResults [hidden-layer output-layer])

(defrecord IterationResults [current-net error-value])

(def test-net (->NeuralNet my-wh my-wo))

(defn identity-matrix-with-one-zero
  [n]
  (diag (utils/n-ones-and-a-zero n)))

(defn backward-pass-output
  [net desired-response fpr]
  (let [current-error-vector (minus desired-response (.output-layer-values (.output-layer fpr)))
        current-del-output (mult current-error-vector (mapv activation-function-deriv (.induced-local-field (.output-layer fpr))))
        delta-W (mult learning-rate (mmult (.hidden-layer-values (.hidden-layer fpr)) (trans current-del-output)))]
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

(defn add-network-weights
  [net1 net2]
  (let [wh-new (plus (.hidden-weights net1) (.hidden-weights net2))
        wo-new (plus (.output-weights net1) (.output-weights net2))]
    (->NeuralNet wh-new wo-new)))

(defn epoch-reducer
  [result1 result2]
  (let [net1 (.current-net result1)
        net2 (.current-net result2)
        error1 (.error-value result1)
        error2 (.error-value result2)
        wh-new (plus (.hidden-weights net1) (.hidden-weights net2))
        wo-new (plus (.output-weights net1) (.output-weights net2))
        new-net (->NeuralNet wh-new wo-new)
        total-error (+ error1 error2)]
    (->IterationResults new-net total-error)))

(defn iteration
  ([net training-example]
   (iteration net (trans (matrix (.input-vector training-example))) (matrix (.desired-response training-example))))

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
   (let [num-examples (length training-set)]
     (loop [current-net net current-err 999.0]
       (let [current-iteration-adjustment (reduce epoch-reducer (map (partial iteration current-net) training-set))
             current-iteration-result (add-network-weights current-net (.current-net current-iteration-adjustment))]
         (let [current-avg-err (/ (.error-value current-iteration-adjustment) num-examples)]
           (do
             (println current-avg-err)
             (if-not (approx-equals? current-avg-err 0.0)
               (recur current-iteration-result current-avg-err)
               current-net))))))))
