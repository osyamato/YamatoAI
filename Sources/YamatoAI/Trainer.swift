import Foundation

struct Trainer {

    let learningRate: Float

    func train(

        embeddings: [EmbeddingVector],

        correctTokenID: Int,

        attention: inout Attention,

        feedForward: inout FeedForward,

        outputLayer: inout OutputLayer

    ) {

        // -------------------------
        // Attention
        // -------------------------

        let attentionOutputs = attention.calculate(

            embeddings: embeddings

        )

        guard let attentionOutput = attentionOutputs.last else {

            return
        }


        // -------------------------
        // FeedForward
        // -------------------------

        let feedForwardOutput = feedForward.calculate(

            vector: attentionOutput

        )


        // -------------------------
        // OutputLayer
        // -------------------------

        let output = outputLayer.calculate(

            vector: feedForwardOutput

        )


        // -------------------------
        // Softmax
        // -------------------------

        let probabilities = Softmax().calculate(

            scores: output.values

        )


        // -------------------------
        // Output Gradient
        // -------------------------

        let outputGradient = OutputGradient().calculate(

            probabilities: probabilities,

            correctTokenID: correctTokenID

        )


        // -------------------------
        // OutputLayer Weight Gradient
        // -------------------------

        let outputLayerWeightGradient =
            OutputLayerGradient().calculate(

                vector: feedForwardOutput,

                outputGradient: outputGradient

            )


        // -------------------------
        // OutputLayer
        // ↓
        // FeedForward
        // -------------------------

        let feedForwardOutputGradient =
            InputGradient().calculate(

                outputGradient: outputGradient,

                weight: outputLayer.weight

            )


        // -------------------------
        // FeedForward W1 / W2 Gradient
        // -------------------------

        let feedForwardWeightGradient =
            FeedForwardWeightGradient().calculate(

                input: attentionOutput,

                outputGradient: feedForwardOutputGradient,

                w1: feedForward.w1,

                w2: feedForward.w2

            )


        // -------------------------
        // FeedForward
        // ↓
        // Attention
        // -------------------------

        let attentionOutputGradient =
            FeedForwardGradient().calculate(

                input: attentionOutput,

                outputGradient: feedForwardOutputGradient,

                w1: feedForward.w1,

                w2: feedForward.w2

            )


        // -------------------------
        // Attention Output Gradients
        // -------------------------
        //
        // 今回は最後のAttention出力だけを
        // 次Token予測に使用している。
        //
        // そのため、それ以外の出力Gradientは0。
        // -------------------------

        var attentionOutputGradients: [EmbeddingVector] = []

        for index in attentionOutputs.indices {

            if index == attentionOutputs.count - 1 {

                attentionOutputGradients.append(

                    attentionOutputGradient

                )

            } else {

                attentionOutputGradients.append(

                    EmbeddingVector(

                        values: Array(

                            repeating: 0,

                            count: attentionOutputs[index].values.count

                        )
                    )
                )
            }
        }


        // -------------------------
        // Attention
        // Wq / Wk / Wv Gradient
        // -------------------------

        let attentionWeightGradient =
            AttentionWeightGradient().calculate(

                embeddings: embeddings,

                outputGradients: attentionOutputGradients,

                wq: attention.wq,

                wk: attention.wk,

                wv: attention.wv

            )


        // =========================
        // Update
        // =========================


        // -------------------------
        // OutputLayer Update
        // -------------------------

        outputLayer.update(

            gradient: outputLayerWeightGradient,

            learningRate: learningRate

        )


        // -------------------------
        // FeedForward Update
        // -------------------------

        feedForward.update(

            w1Gradient:
                feedForwardWeightGradient.w1Gradient,

            w2Gradient:
                feedForwardWeightGradient.w2Gradient,

            learningRate: learningRate

        )


        // -------------------------
        // Attention Update
        // -------------------------

        attention.update(

            wqGradient:
                attentionWeightGradient.wqGradient,

            wkGradient:
                attentionWeightGradient.wkGradient,

            wvGradient:
                attentionWeightGradient.wvGradient,

            learningRate: learningRate

        )
    }
}
