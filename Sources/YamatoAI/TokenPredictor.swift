import Foundation

struct TokenPredictor {

    let outputLayer: OutputLayer

    func calculate(

        vector: EmbeddingVector

    ) -> [Float] {

        let output = outputLayer.calculate(

            vector: vector

        )

        let probabilities = Softmax().calculate(

            scores: output.values

        )

        return probabilities
    }
}
