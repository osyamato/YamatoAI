import Foundation

struct DotProduct {

    func calculate(

        between first: EmbeddingVector,

        and second: EmbeddingVector

    ) -> Float {

        let firstValues = first.values

        let secondValues = second.values

        guard firstValues.count == secondValues.count else {

            return 0

        }
        

        var dotProduct: Float = 0
        
        for index in firstValues.indices {

            let product =

                firstValues[index] * secondValues[index]

            dotProduct += product

        }
        
        

        return dotProduct

    }

}
