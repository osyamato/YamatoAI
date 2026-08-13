import Foundation

struct TransformerBlockGradient {

    func calculate(

        embeddings: [EmbeddingVector],

        outputGradients: [EmbeddingVector],

        multiHeadAttention: MultiHeadAttention,

        feedForward: FeedForward

    ) -> TransformerBlockGradientResult {

        // =====================================
        // Forward
        // =====================================


        // -------------------------
        // MultiHeadAttention
        // -------------------------

        let attentionResults =
            multiHeadAttention.calculate(

                embeddings: embeddings

            )


        // -------------------------
        // Residual 1
        // -------------------------

        var residualResults: [EmbeddingVector] = []

        for index in embeddings.indices {

            let residual =
                ResidualConnection().calculate(

                    original:
                        embeddings[index],

                    transformed:
                        attentionResults[index]

                )

            residualResults.append(
                residual
            )
        }


        // -------------------------
        // LayerNorm 1
        // -------------------------

        var normalizedResults: [EmbeddingVector] = []

        for residual in residualResults {

            let normalized =
                LayerNormalization().calculate(

                    vector: residual

                )

            normalizedResults.append(
                normalized
            )
        }


        // -------------------------
        // FeedForward
        // -------------------------

        var feedForwardResults: [EmbeddingVector] = []

        for normalized in normalizedResults {

            let result =
                feedForward.calculate(

                    vector: normalized

                )

            feedForwardResults.append(
                result
            )
        }


        // -------------------------
        // Residual 2
        // -------------------------

        var secondResidualResults: [EmbeddingVector] = []

        for index in normalizedResults.indices {

            let residual =
                ResidualConnection().calculate(

                    original:
                        normalizedResults[index],

                    transformed:
                        feedForwardResults[index]

                )

            secondResidualResults.append(
                residual
            )
        }


        // =====================================
        // Backward
        // =====================================


        // -------------------------
        // FeedForward
        // W1 Gradient 初期化
        // -------------------------

        var w1GradientValues =
            feedForward.w1.values.map {

                Array(

                    repeating: Float(0),

                    count: $0.count

                )
            }


        // -------------------------
        // FeedForward
        // W2 Gradient 初期化
        // -------------------------

        var w2GradientValues =
            feedForward.w2.values.map {

                Array(

                    repeating: Float(0),

                    count: $0.count

                )
            }


        // -------------------------
        // MultiHeadAttentionへ
        // 戻すGradient
        // -------------------------

        var attentionOutputGradients: [EmbeddingVector] = []


        // -------------------------
        // Residual 1から
        // Embeddingへ直接戻るGradient
        // -------------------------

        var directInputGradients: [EmbeddingVector] = []


        // =====================================
        // Tokenごとに逆伝播
        // =====================================

        for index in outputGradients.indices {


            // -------------------------
            // LayerNorm 2
            // -------------------------

            let secondLayerNormGradient =
                LayerNormalizationGradient().calculate(

                    input:
                        secondResidualResults[index],

                    outputGradient:
                        outputGradients[index]

                )


            // -------------------------
            // Residual 2
            //
            // normalized
            //      ↓
            //      + ---- FeedForward
            //      ↓
            // LayerNorm 2
            // -------------------------

            let secondResidualGradient =
                ResidualGradient().calculate(

                    outputGradient:
                        secondLayerNormGradient

                )


            // -------------------------
            // FeedForward
            // W1 / W2 Gradient
            // -------------------------

            let feedForwardWeightGradient =
                FeedForwardWeightGradient().calculate(

                    input:
                        normalizedResults[index],

                    outputGradient:
                        secondResidualGradient
                            .branchGradient,

                    w1:
                        feedForward.w1,

                    w2:
                        feedForward.w2

                )


            // -------------------------
            // W1 Gradient 加算
            // -------------------------

            for rowIndex in w1GradientValues.indices {

                for columnIndex
                    in w1GradientValues[rowIndex].indices {

                    w1GradientValues[rowIndex][columnIndex] +=

                        feedForwardWeightGradient
                            .w1Gradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // W2 Gradient 加算
            // -------------------------

            for rowIndex in w2GradientValues.indices {

                for columnIndex
                    in w2GradientValues[rowIndex].indices {

                    w2GradientValues[rowIndex][columnIndex] +=

                        feedForwardWeightGradient
                            .w2Gradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // FeedForward
            // Input Gradient
            // -------------------------

            let feedForwardInputGradient =
                FeedForwardGradient().calculate(

                    input:
                        normalizedResults[index],

                    outputGradient:
                        secondResidualGradient
                            .branchGradient,

                    w1:
                        feedForward.w1,

                    w2:
                        feedForward.w2

                )


            // -------------------------
            // Residual 2
            //
            // 直通Gradient
            // +
            // FeedForward経由Gradient
            // -------------------------

            var normalizedGradientValues: [Float] = []

            for valueIndex
                in normalizedResults[index].values.indices {

                let directGradient =
                    secondResidualGradient
                        .inputGradient
                        .values[valueIndex]

                let feedForwardGradient =
                    feedForwardInputGradient
                        .values[valueIndex]


                normalizedGradientValues.append(

                    directGradient
                    +
                    feedForwardGradient

                )
            }


            let normalizedGradient =
                EmbeddingVector(

                    values:
                        normalizedGradientValues

                )


            // -------------------------
            // LayerNorm 1
            // -------------------------

            let firstLayerNormGradient =
                LayerNormalizationGradient().calculate(

                    input:
                        residualResults[index],

                    outputGradient:
                        normalizedGradient

                )


            // -------------------------
            // Residual 1
            //
            // Embedding
            //     ↓
            //     + ---- MultiHeadAttention
            //     ↓
            // LayerNorm 1
            // -------------------------

            let firstResidualGradient =
                ResidualGradient().calculate(

                    outputGradient:
                        firstLayerNormGradient

                )


            // -------------------------
            // MultiHeadAttention側へ
            // -------------------------

            attentionOutputGradients.append(

                firstResidualGradient
                    .branchGradient

            )


            // -------------------------
            // Embeddingへの直通Gradient
            // -------------------------

            directInputGradients.append(

                firstResidualGradient
                    .inputGradient

            )
        }


        // =====================================
        // MultiHeadAttention
        // Backward
        // =====================================

        let multiHeadGradient =
            MultiHeadAttentionGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    attentionOutputGradients,

                heads:
                    multiHeadAttention.heads,

                wo:
                    multiHeadAttention.wo

            )


        // =====================================
        // TransformerBlock
        // Input Gradient
        //
        // Residual直通
        // +
        // MultiHeadAttention経由
        // =====================================

        var inputGradients: [EmbeddingVector] = []

        for index in embeddings.indices {

            var gradientValues: [Float] = []

            for valueIndex
                in embeddings[index].values.indices {

                let directGradient =
                    directInputGradients[index]
                        .values[valueIndex]

                let attentionGradient =
                    multiHeadGradient
                        .inputGradients[index]
                        .values[valueIndex]


                gradientValues.append(

                    directGradient
                    +
                    attentionGradient

                )
            }


            inputGradients.append(

                EmbeddingVector(

                    values:
                        gradientValues

                )
            )
        }


        // =====================================
        // Result
        // =====================================

        return TransformerBlockGradientResult(

            inputGradients:
                inputGradients,

            multiHeadAttentionGradient:
                multiHeadGradient,

            w1Gradient:
                Matrix(

                    values:
                        w1GradientValues

                ),

            w2Gradient:
                Matrix(

                    values:
                        w2GradientValues

                )
        )
    }
}
