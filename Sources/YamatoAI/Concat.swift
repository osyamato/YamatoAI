import Foundation

struct Concat {

    func calculate(

        headResults: [[EmbeddingVector]]

    ) -> [EmbeddingVector] {

        var results: [EmbeddingVector] = []
        
        for tokenIndex in headResults[0].indices {

            var concatenatedValues: [Float] = []

            for headIndex in headResults.indices {

                concatenatedValues.append(

                    contentsOf: headResults[headIndex][tokenIndex].values

                )

            }

            let concatenatedVector = EmbeddingVector(

                values: concatenatedValues

            )

            results.append(concatenatedVector)

        }

        return results

    }

}
