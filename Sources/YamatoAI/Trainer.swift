import Foundation

struct Trainer {

    let learningRate: Float


    // =====================================
    // Legacy Training
    //
    // Attention
    // ↓
    // FeedForward
    // ↓
    // OutputLayer
    //
    // 既存テスト互換用
    // =====================================

    func train(

        embeddings: [EmbeddingVector],

        correctTokenID: Int,

        attention: inout Attention,

        feedForward: inout FeedForward,

        outputLayer: inout OutputLayer

    ) {

        // =====================================
        // Attention Forward
        // =====================================

        let attentionOutputs =
            attention.calculate(

                embeddings:
                    embeddings

            )


        guard let attentionOutput =
                attentionOutputs.last else {

            return
        }


        // =====================================
        // FeedForward Forward
        // =====================================

        let feedForwardOutput =
            feedForward.calculate(

                vector:
                    attentionOutput

            )


        // =====================================
        // OutputLayer Forward
        // =====================================

        let output =
            outputLayer.calculate(

                vector:
                    feedForwardOutput

            )


        // =====================================
        // Softmax
        // =====================================

        let probabilities =
            Softmax().calculate(

                scores:
                    output.values

            )


        // =====================================
        // Output Gradient
        // =====================================

        let outputGradient =
            OutputGradient().calculate(

                probabilities:
                    probabilities,

                correctTokenID:
                    correctTokenID

            )


        // =====================================
        // OutputLayer Weight Gradient
        // =====================================

        let outputLayerWeightGradient =
            OutputLayerGradient().calculate(

                vector:
                    feedForwardOutput,

                outputGradient:
                    outputGradient

            )


        // =====================================
        // OutputLayer
        // ↓
        // FeedForward
        // =====================================

        let feedForwardOutputGradient =
            InputGradient().calculate(

                outputGradient:
                    outputGradient,

                weight:
                    outputLayer.weight

            )


        // =====================================
        // FeedForward Weight Gradient
        // =====================================

        let feedForwardWeightGradient =
            FeedForwardWeightGradient().calculate(

                input:
                    attentionOutput,

                outputGradient:
                    feedForwardOutputGradient,

                w1:
                    feedForward.w1,

                w2:
                    feedForward.w2

            )


        // =====================================
        // FeedForward
        // ↓
        // Attention
        // =====================================

        let attentionOutputGradient =
            FeedForwardGradient().calculate(

                input:
                    attentionOutput,

                outputGradient:
                    feedForwardOutputGradient,

                w1:
                    feedForward.w1,

                w2:
                    feedForward.w2

            )


        // =====================================
        // Attention Output Gradients
        //
        // 最後のTokenだけを
        // 次Token予測に使用
        // =====================================

        var attentionOutputGradients:
            [EmbeddingVector] = []


        for index in attentionOutputs.indices {

            if index ==
                attentionOutputs.count - 1 {

                attentionOutputGradients.append(

                    attentionOutputGradient

                )

            } else {

                attentionOutputGradients.append(

                    EmbeddingVector(

                        values:
                            Array(

                                repeating: 0,

                                count:
                                    attentionOutputs[index]
                                        .values
                                        .count

                            )
                    )
                )
            }
        }


        // =====================================
        // Attention Weight Gradient
        // =====================================

        let attentionWeightGradient =
            AttentionWeightGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    attentionOutputGradients,

                wq:
                    attention.wq,

                wk:
                    attention.wk,

                wv:
                    attention.wv

            )


        // =====================================
        // Update
        // =====================================


        // -------------------------------------
        // OutputLayer
        // -------------------------------------

        outputLayer.update(

            gradient:
                outputLayerWeightGradient,

            learningRate:
                learningRate

        )


        // -------------------------------------
        // FeedForward
        // -------------------------------------

        feedForward.update(

            w1Gradient:
                feedForwardWeightGradient.w1Gradient,

            w2Gradient:
                feedForwardWeightGradient.w2Gradient,

            learningRate:
                learningRate

        )


        // -------------------------------------
        // Attention
        // -------------------------------------

        attention.update(

            wqGradient:
                attentionWeightGradient.wqGradient,

            wkGradient:
                attentionWeightGradient.wkGradient,

            wvGradient:
                attentionWeightGradient.wvGradient,

            learningRate:
                learningRate

        )
    }



    // =====================================
    // Transformer Training
    //
    // Embedding
    // ↓
    // Transformer
    // ↓
    // OutputLayer
    // ↓
    // Softmax
    // ↓
    // Backward
    // ↓
    // Update
    // =====================================

    func train(

        embeddings: [EmbeddingVector],

        correctTokenID: Int,

        transformer: inout Transformer,

        outputLayer: inout OutputLayer

    ) {

        // =====================================
        // Transformer Forward
        // =====================================

        let transformerOutputs =
            transformer.calculate(

                embeddings:
                    embeddings

            )


        guard let transformerOutput =
                transformerOutputs.last else {

            return
        }


        // =====================================
        // OutputLayer Forward
        // =====================================

        let output =
            outputLayer.calculate(

                vector:
                    transformerOutput

            )


        // =====================================
        // Softmax
        // =====================================

        let probabilities =
            Softmax().calculate(

                scores:
                    output.values

            )


        // =====================================
        // Output Gradient
        //
        // softmax + cross entropy
        //
        // dL / dLogits
        // =====================================

        let outputGradient =
            OutputGradient().calculate(

                probabilities:
                    probabilities,

                correctTokenID:
                    correctTokenID

            )


        // =====================================
        // OutputLayer Weight Gradient
        //
        // dL / dWo
        // =====================================

        let outputLayerWeightGradient =
            OutputLayerGradient().calculate(

                vector:
                    transformerOutput,

                outputGradient:
                    outputGradient

            )


        // =====================================
        // OutputLayer Input Gradient
        //
        // OutputLayer
        // ↓
        // Transformer
        //
        // dL / dTransformerOutput
        // =====================================

        let transformerLastOutputGradient =
            InputGradient().calculate(

                outputGradient:
                    outputGradient,

                weight:
                    outputLayer.weight

            )


        // =====================================
        // Transformer Output Gradients
        //
        // 現在のYamatoAIでは
        //
        // Transformerの最後のToken出力
        // ↓
        // OutputLayer
        // ↓
        // 次Token予測
        //
        // としている。
        //
        // したがって直接Lossが入るのは
        // 最後のTokenのみ。
        //
        // Attention内部では
        // ここから他TokenへGradientが流れる。
        // =====================================

        var transformerOutputGradients:
            [EmbeddingVector] = []


        for index in transformerOutputs.indices {

            if index ==
                transformerOutputs.count - 1 {

                transformerOutputGradients.append(

                    transformerLastOutputGradient

                )

            } else {

                transformerOutputGradients.append(

                    EmbeddingVector(

                        values:
                            Array(

                                repeating: 0,

                                count:
                                    transformerOutputs[index]
                                        .values
                                        .count

                            )
                    )
                )
            }
        }


        // =====================================
        // Transformer Backward
        //
        // 最終Block
        // ↓
        // ...
        // ↓
        // 最初のBlock
        //
        // まで逆伝播
        // =====================================

        let transformerGradient =
            TransformerGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    transformerOutputGradients,

                transformer:
                    transformer

            )


        // =====================================
        // Safety Check
        //
        // NaN / Infinityが発生した状態で
        // Weightを壊さないための最低限の確認
        // =====================================

        for gradientVector
            in transformerGradient.inputGradients {

            for value
                in gradientVector.values {

                guard value.isFinite else {

                    return
                }
            }
        }


        for value
            in outputGradient {

            guard value.isFinite else {

                return
            }
        }


        // =====================================
        // Update
        //
        // Gradient計算はUpdate前のWeightで
        // すべて完了している。
        //
        // その後まとめて更新する。
        // =====================================


        // -------------------------------------
        // OutputLayer Update
        // -------------------------------------

        outputLayer.update(

            gradient:
                outputLayerWeightGradient,

            learningRate:
                learningRate

        )


        // -------------------------------------
        // Transformer Update
        //
        // Block 0
        // Block 1
        // ...
        //
        // 各Block内部の
        //
        // MultiHeadAttention
        // Wq / Wk / Wv / Wo
        //
        // FeedForward
        // W1 / W2
        //
        // を更新
        // -------------------------------------

        transformer.update(

            gradient:
                transformerGradient,

            learningRate:
                learningRate

        )
    }
}
