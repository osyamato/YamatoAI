import Foundation

struct AttentionWeightGradient {

    func calculate(

        embeddings: [EmbeddingVector],

        outputGradients: [EmbeddingVector],

        wq: Matrix,

        wk: Matrix,

        wv: Matrix

    ) -> AttentionWeightGradientResult {

        // =====================================
        // Forwardを再計算
        // =====================================


        // -------------------------
        // Query / Key / Value
        // -------------------------

        var queries: [EmbeddingVector] = []
        var keys: [EmbeddingVector] = []
        var values: [EmbeddingVector] = []

        for embedding in embeddings {

            let query = Query().make(

                from: embedding,

                using: wq

            )

            let key = Key().make(

                from: embedding,

                using: wk

            )

            let value = Value().make(

                from: embedding,

                using: wv

            )

            queries.append(query)

            keys.append(key)

            values.append(value)
        }


        // =====================================
        // Gradientを入れる箱
        // =====================================


        // -------------------------
        // Query Gradient
        // -------------------------

        var queryGradients: [EmbeddingVector] =
            queries.map {

                EmbeddingVector(

                    values: Array(

                        repeating: Float(0),

                        count: $0.values.count

                    )
                )
            }


        // -------------------------
        // Key Gradient
        // -------------------------

        var keyGradients: [EmbeddingVector] =
            keys.map {

                EmbeddingVector(

                    values: Array(

                        repeating: Float(0),

                        count: $0.values.count

                    )
                )
            }


        // -------------------------
        // Value Gradient
        // -------------------------

        var valueGradients: [EmbeddingVector] =
            values.map {

                EmbeddingVector(

                    values: Array(

                        repeating: Float(0),

                        count: $0.values.count

                    )
                )
            }


        // =====================================
        // Attention逆伝播
        // =====================================

        for queryIndex in queries.indices {

            let query = queries[queryIndex]

            let dimension = query.values.count

            var scores: [Float] = []


            // -------------------------
            // Scoreを再計算
            // -------------------------

            for key in keys {

                let score =
                    DotProduct().calculate(

                        between: query,

                        and: key

                    )

                let scaledScore =
                    score / sqrt(Float(dimension))

                scores.append(

                    scaledScore

                )
            }


            // -------------------------
            // Softmax
            // -------------------------

            let probabilities =
                Softmax().calculate(

                    scores: scores

                )


            // -------------------------
            // WeightedValue
            // 逆伝播
            // -------------------------

            let weightedGradient =
                WeightedValueGradient().calculate(

                    probabilities: probabilities,

                    values: values,

                    outputGradient:
                        outputGradients[queryIndex]

                )


            // -------------------------
            // Value Gradientを加算
            // -------------------------

            for valueIndex in values.indices {

                var newValues =
                    valueGradients[valueIndex].values

                for dimensionIndex in newValues.indices {

                    newValues[dimensionIndex] +=

                        weightedGradient
                            .valueGradients[valueIndex]
                            .values[dimensionIndex]
                }

                valueGradients[valueIndex] =
                    EmbeddingVector(

                        values: newValues

                    )
            }


            // -------------------------
            // Softmax
            // 逆伝播
            // -------------------------

            let scaledScoreGradients =
                SoftmaxGradient().calculate(

                    probabilities: probabilities,

                    outputGradient:
                        weightedGradient
                            .probabilityGradients

                )


            // -------------------------
            // ScaledScore
            // ↓
            // DotProduct
            // -------------------------

            for keyIndex in keys.indices {

                let scoreGradient =
                    ScaledScoreGradient().calculate(

                        scaledScoreGradient:
                            scaledScoreGradients[keyIndex],

                        dimension:
                            dimension

                    )


                let dotGradient =
                    DotProductGradient().calculate(

                        query:
                            query,

                        key:
                            keys[keyIndex],

                        scoreGradient:
                            scoreGradient

                    )


                // -------------------------
                // Query Gradientを加算
                // -------------------------

                var newQueryValues =
                    queryGradients[queryIndex].values

                for valueIndex in newQueryValues.indices {

                    newQueryValues[valueIndex] +=

                        dotGradient
                            .queryGradient
                            .values[valueIndex]
                }

                queryGradients[queryIndex] =
                    EmbeddingVector(

                        values:
                            newQueryValues

                    )


                // -------------------------
                // Key Gradientを加算
                // -------------------------

                var newKeyValues =
                    keyGradients[keyIndex].values

                for valueIndex in newKeyValues.indices {

                    newKeyValues[valueIndex] +=

                        dotGradient
                            .keyGradient
                            .values[valueIndex]
                }

                keyGradients[keyIndex] =
                    EmbeddingVector(

                        values:
                            newKeyValues

                    )
            }
        }


        // =====================================
        // Weight Gradient
        // =====================================


        // -------------------------
        // Wq Gradient
        // -------------------------

        var wqGradientValues =
            wq.values.map {

                Array(

                    repeating: Float(0),

                    count: $0.count

                )
            }


        // -------------------------
        // Wk Gradient
        // -------------------------

        var wkGradientValues =
            wk.values.map {

                Array(

                    repeating: Float(0),

                    count: $0.count

                )
            }


        // -------------------------
        // Wv Gradient
        // -------------------------

        var wvGradientValues =
            wv.values.map {

                Array(

                    repeating: Float(0),

                    count: $0.count

                )
            }


        // -------------------------
        // Embedding × Gradient
        // -------------------------

        for embeddingIndex in embeddings.indices {

            let embedding =
                embeddings[embeddingIndex]


            // -------------------------
            // Wq
            // -------------------------

            let currentWqGradient =
                OutputLayerGradient().calculate(

                    vector:
                        embedding,

                    outputGradient:
                        queryGradients[embeddingIndex]
                            .values

                )

            for rowIndex in wqGradientValues.indices {

                for columnIndex
                    in wqGradientValues[rowIndex].indices {

                    wqGradientValues[rowIndex][columnIndex] +=

                        currentWqGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // Wk
            // -------------------------

            let currentWkGradient =
                OutputLayerGradient().calculate(

                    vector:
                        embedding,

                    outputGradient:
                        keyGradients[embeddingIndex]
                            .values

                )

            for rowIndex in wkGradientValues.indices {

                for columnIndex
                    in wkGradientValues[rowIndex].indices {

                    wkGradientValues[rowIndex][columnIndex] +=

                        currentWkGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // Wv
            // -------------------------

            let currentWvGradient =
                OutputLayerGradient().calculate(

                    vector:
                        embedding,

                    outputGradient:
                        valueGradients[embeddingIndex]
                            .values

                )

            for rowIndex in wvGradientValues.indices {

                for columnIndex
                    in wvGradientValues[rowIndex].indices {

                    wvGradientValues[rowIndex][columnIndex] +=

                        currentWvGradient
                            .values[rowIndex][columnIndex]
                }
            }
        }


        // =====================================
        // Input Gradient
        //
        // Query / Key / Value
        // ↓
        // 元のEmbeddingへ戻す
        // =====================================

        var inputGradients: [EmbeddingVector] = []

        for embeddingIndex in embeddings.indices {


            // -------------------------
            // Query
            // ↓
            // Embedding
            // -------------------------

            let queryInputGradient =
                InputGradient().calculate(

                    outputGradient:
                        queryGradients[embeddingIndex]
                            .values,

                    weight:
                        wq

                )


            // -------------------------
            // Key
            // ↓
            // Embedding
            // -------------------------

            let keyInputGradient =
                InputGradient().calculate(

                    outputGradient:
                        keyGradients[embeddingIndex]
                            .values,

                    weight:
                        wk

                )


            // -------------------------
            // Value
            // ↓
            // Embedding
            // -------------------------

            let valueInputGradient =
                InputGradient().calculate(

                    outputGradient:
                        valueGradients[embeddingIndex]
                            .values,

                    weight:
                        wv

                )


            // -------------------------
            // 3経路を合流
            // -------------------------

            var gradientValues: [Float] = []

            for valueIndex
                in embeddings[embeddingIndex]
                    .values.indices {

                let gradient =

                    queryInputGradient
                        .values[valueIndex]

                    +

                    keyInputGradient
                        .values[valueIndex]

                    +

                    valueInputGradient
                        .values[valueIndex]


                gradientValues.append(

                    gradient

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

        return AttentionWeightGradientResult(

            inputGradients:
                inputGradients,

            wqGradient:
                Matrix(

                    values:
                        wqGradientValues

                ),

            wkGradient:
                Matrix(

                    values:
                        wkGradientValues

                ),

            wvGradient:
                Matrix(

                    values:
                        wvGradientValues

                )
        )
    }
}
