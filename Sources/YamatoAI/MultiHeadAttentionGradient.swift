import Foundation

struct MultiHeadAttentionGradient {

    func calculate(

        embeddings: [EmbeddingVector],

        outputGradients: [EmbeddingVector],

        heads: [MultiHead],

        wo: Matrix

    ) -> MultiHeadAttentionGradientResult {

        // =====================================
        // Forwardを再計算
        // =====================================


        // -------------------------
        // 各HeadのAttention結果
        // -------------------------

        var allHeadResults: [[EmbeddingVector]] = []

        for head in heads {

            let attention = Attention(

                wq: head.wq,

                wk: head.wk,

                wv: head.wv

            )

            let results = attention.calculate(

                embeddings: embeddings

            )

            allHeadResults.append(

                results

            )
        }


        // -------------------------
        // Concat
        // -------------------------

        let concatenated = Concat().calculate(

            headResults: allHeadResults

        )


        // =====================================
        // Gradient用の箱を初期化
        // =====================================


        // -------------------------
        // Wo Gradient
        // -------------------------

        var woGradientValues = Array(

            repeating: Array(

                repeating: Float(0),

                count: wo.values[0].count

            ),

            count: wo.values.count

        )


        // -------------------------
        // 各Headへ戻すGradient
        // -------------------------

        var headOutputGradients: [[EmbeddingVector]] = Array(

            repeating: [],

            count: heads.count

        )


        // =====================================
        // OutputProjection
        // ↓
        // Concat
        // ↓
        // 各Head
        // =====================================

        for index in outputGradients.indices {


            // -------------------------
            // OutputProjection Gradient
            // -------------------------

            let projectionGradient =
                OutputProjectionGradient().calculate(

                    input:
                        concatenated[index],

                    outputGradient:
                        outputGradients[index],

                    weight:
                        wo

                )


            // -------------------------
            // Wo Gradient 加算
            // -------------------------

            for rowIndex in woGradientValues.indices {

                for columnIndex
                    in woGradientValues[rowIndex].indices {

                    woGradientValues[rowIndex][columnIndex] +=

                        projectionGradient
                            .weightGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // Concat Gradient
            //
            // 1本のGradientを
            // 各Headへ分割
            // -------------------------

            let splitGradients =
                ConcatGradient().calculate(

                    outputGradient:
                        projectionGradient.inputGradient,

                    headCount:
                        heads.count

                )


            // -------------------------
            // 各HeadへGradientを保存
            // -------------------------

            for headIndex in heads.indices {

                headOutputGradients[headIndex].append(

                    splitGradients[headIndex]

                )
            }
        }


        // =====================================
        // 各Attention Headを逆伝播
        // =====================================

        var headGradients:
            [AttentionWeightGradientResult] = []

        for headIndex in heads.indices {

            let head =
                heads[headIndex]


            let gradient =
                AttentionWeightGradient().calculate(

                    embeddings:
                        embeddings,

                    outputGradients:
                        headOutputGradients[headIndex],

                    wq:
                        head.wq,

                    wk:
                        head.wk,

                    wv:
                        head.wv

                )


            headGradients.append(

                gradient

            )
        }


        // =====================================
        // 各HeadのInput Gradientを合流
        //
        // Head 0
        // Head 1
        // Head 2
        // ...
        //
        // ↓
        //
        // 元のEmbedding Gradient
        // =====================================

        var inputGradientValues =
            embeddings.map {

                Array(

                    repeating: Float(0),

                    count: $0.values.count

                )
            }


        for headGradient in headGradients {

            for embeddingIndex in embeddings.indices {

                for valueIndex
                    in inputGradientValues[embeddingIndex].indices {

                    inputGradientValues[embeddingIndex][valueIndex] +=

                        headGradient
                            .inputGradients[embeddingIndex]
                            .values[valueIndex]
                }
            }
        }


        // -------------------------
        // EmbeddingVectorへ変換
        // -------------------------

        var inputGradients: [EmbeddingVector] = []

        for values in inputGradientValues {

            inputGradients.append(

                EmbeddingVector(

                    values: values

                )
            )
        }


        // =====================================
        // Result
        // =====================================

        return MultiHeadAttentionGradientResult(

            inputGradients:
                inputGradients,

            woGradient:
                Matrix(

                    values:
                        woGradientValues

                ),

            headGradients:
                headGradients

        )
    }
}
