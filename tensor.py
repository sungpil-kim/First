# 텐서플로어 기초 강의 요약

## 텐서플로어 내용
import tensorflow as tf

텐서 = tf.constant([3,4,5])
텐서2 = tf.constant([6,7,8])
텐서3 = tf.zeros([3,2,3])
텐서4 = tf.zeros([1,2,3],
                 [4,5,6])

print(텐서+텐서2) #[9,11,13]

print(텐서3) #[[[0,0,0]]] [[[0,0,0]]] [[0,0,0]]]

print(텐서3.shape) #(1,3)

print(텐서4.shape) #(2,3)
