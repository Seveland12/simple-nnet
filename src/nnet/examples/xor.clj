(ns nnet.examples.xor
  (:require [nnet.data-structures :refer :all]
            [clojure.core.matrix :as m]))

(def training-set-xor [(->TrainingExample [[-0.5 -0.5 1.0]] [[-0.5]])
                       (->TrainingExample [[-0.5 0.5 1.0]] [[0.5]])
                       (->TrainingExample [[0.5 -0.5 1.0]] [[0.5]])
                       (->TrainingExample [[0.5 0.5 1.0]] [[-0.5]])])

(def output-mean-vector [[0.5]])

(defn response-interpretation
  [output-vector]
  (let [mean-added (m/add output-vector output-mean-vector)]
    mean-added))
