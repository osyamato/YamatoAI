import Foundation

struct MultiHeadAttention {

    var heads: [MultiHead]

    var wo: Matrix


    func calculate(

        embeddings: [EmbeddingVector]

    ) -> [EmbeddingVector] {

        var allHeadResults: [[EmbeddingVector]] = []

        for head in heads {

            let attention = Attention(

                wq: head.wq,

                wk: head.wk,

                wv: head.wv

            )

            let result = attention.calculate(

                embeddings: embeddings

            )

            allHeadResults.append(result)
        }


        let concatenated = Concat().calculate(

            headResults: allHeadResults

        )


        var results: [EmbeddingVector] = []

        for vector in concatenated {

            let projectedVector = OutputProjection().calculate(

                vector: vector,

                using: wo

            )

            results.append(projectedVector)
        }

        return results
    }


    // MARK: - Update

    mutating func update(

        gradient: MultiHeadAttentionGradientResult,

        learningRate: Float

    ) {

        // -------------------------
        // Wo Update
        // -------------------------

        var newWoValues = wo.values

        for rowIndex in wo.values.indices {

            for columnIndex in wo.values[rowIndex].indices {

                newWoValues[rowIndex][columnIndex] -=

                    learningRate
                    * gradient
                        .woGradient
                        .values[rowIndex][columnIndex]
            }
        }

        wo = Matrix(

            values: newWoValues

        )


        // -------------------------
        // Heads Update
        // -------------------------

        var newHeads: [MultiHead] = []

        for headIndex in heads.indices {

            let head = heads[headIndex]

            let headGradient =
                gradient.headGradients[headIndex]


            // -------------------------
            // Wq
            // -------------------------

            var newWqValues = head.wq.values

            for rowIndex in head.wq.values.indices {

                for columnIndex in head.wq.values[rowIndex].indices {

                    newWqValues[rowIndex][columnIndex] -=

                        learningRate
                        * headGradient
                            .wqGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // Wk
            // -------------------------

            var newWkValues = head.wk.values

            for rowIndex in head.wk.values.indices {

                for columnIndex in head.wk.values[rowIndex].indices {

                    newWkValues[rowIndex][columnIndex] -=

                        learningRate
                        * headGradient
                            .wkGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // Wv
            // -------------------------

            var newWvValues = head.wv.values

            for rowIndex in head.wv.values.indices {

                for columnIndex in head.wv.values[rowIndex].indices {

                    newWvValues[rowIndex][columnIndex] -=

                        learningRate
                        * headGradient
                            .wvGradient
                            .values[rowIndex][columnIndex]
                }
            }


            // -------------------------
            // New Head
            // -------------------------

            let newHead = MultiHead(

                wq: Matrix(
                    values: newWqValues
                ),

                wk: Matrix(
                    values: newWkValues
                ),

                wv: Matrix(
                    values: newWvValues
                )
            )

            newHeads.append(newHead)
        }


        heads = newHeads
    }
}
