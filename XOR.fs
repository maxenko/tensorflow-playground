namespace tftest

open System
open System.IO
open System.Collections.Generic
open System.Text

open Tensorflow
open Tensorflow.Keras
open Tensorflow.Keras.Layers
open Tensorflow.Keras.ArgsDefinition
open Tensorflow.Keras.Models
open Tensorflow.Keras.Optimizers
open Tensorflow.Keras.Losses
open Tensorflow.Keras.Metrics
open Tensorflow.NumPy

open type Tensorflow.KerasApi

module XOR =
    let tf = Binding.tf

    let layers = new LayersApi()
  
    let Run() =

        let x = np.array( array2D [|
            [|0f;0f|]
            [|0f;1f|]
            [|1f;0f|]
            [|1f;1f|]
        |])

        let y = np.array(
            [|
                0f
                1f
                1f
                0f
            |]
        )
        
        // note: will learn faster with more tensors in the hidden layers
        let input       = layers.Input(2)
        let h_dense1    = layers.Dense(2, "sigmoid").Apply(input)
        let h_dense2    = layers.Dense(2, "sigmoid").Apply(h_dense1)
        let output      = layers.Dense(1, "sigmoid").Apply(h_dense2)
        
        let model = keras.Model(input, output, name="XOR")
        
        model.summary()

        model.compile(
            keras.optimizers.Adam(0.02f),
            keras.losses.MeanSquaredError(),
            metrics=[|"accuracy"|]
            )

        model.fit(x, y, epochs=400, batch_size=1, verbose=1, use_multiprocessing=true)
        model.save("xor.h5")

        ()

