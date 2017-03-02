(ns nnet.examples.xor
  (:require [nnet.data-structures :refer :all]
            [clojure.core.matrix :as m]))

(def training-set-xor [(->TrainingExample [[-0.5 -0.5 1.0]] [[-0.5]])
                       (->TrainingExample [[-0.5 0.5 1.0]] [[0.5]])
                       (->TrainingExample [[0.5 -0.5 1.0]] [[0.5]])
                       (->TrainingExample [[0.5 0.5 1.0]] [[-0.5]])])

(def xor-output-mean-vector [[0.5]])

(defn xor-response-interpretation
  [output-vector]
  (let [mean-added (m/add output-vector xor-output-mean-vector)]
    mean-added))
