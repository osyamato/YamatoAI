import Foundation

struct TransformerBlock {

    var multiHeadAttention: MultiHeadAttention

    var feedForward: FeedForward


    // =====================================
    // Forward
    // =====================================

    func calculate(

        embeddings: [EmbeddingVector]

    ) -> [EmbeddingVector] {

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


        // -------------------------
        // LayerNorm 2
        // -------------------------

        var finalResults: [EmbeddingVector] = []

        for residual in secondResidualResults {

            let normalized =
                LayerNormalization().calculate(

                    vector: residual

                )

            finalResults.append(
                normalized
            )
        }


        return finalResults
    }


    // =====================================
    // Update
    // =====================================

    mutating func update(

        gradient: TransformerBlockGradientResult,

        learningRate: Float

    ) {

        // -------------------------
        // MultiHeadAttention
        // -------------------------

        multiHeadAttention.update(

            gradient:
                gradient.multiHeadAttentionGradient,

            learningRate:
                learningRate

        )


        // -------------------------
        // FeedForward
        // -------------------------

        feedForward.update(

            w1Gradient:
                gradient.w1Gradient,

            w2Gradient:
                gradient.w2Gradient,

            learningRate:
                learningRate

        )
    }
}
