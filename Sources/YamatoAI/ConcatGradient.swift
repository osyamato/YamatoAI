import Foundation

struct ConcatGradient {

    func calculate(

        outputGradient: EmbeddingVector,

        headCount: Int

    ) -> [EmbeddingVector] {

        guard headCount > 0 else {

            return []
        }

        let headDimension =
            outputGradient.values.count / headCount

        var results: [EmbeddingVector] = []

        for headIndex in 0..<headCount {

            let startIndex =
                headIndex * headDimension

            let endIndex =
                startIndex + headDimension

            let values = Array(

                outputGradient.values[
                    startIndex..<endIndex
                ]
            )

            results.append(

                EmbeddingVector(
                    values: values
                )
            )
        }

        return results
    }
}
