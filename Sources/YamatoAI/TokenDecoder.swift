import Foundation

struct TokenDecoder {

    let predictor: TokenPredictor

    let selector: TokenSelector

    let vocabulary: Vocabulary

    func decode(

        vector: EmbeddingVector

    ) -> String? {

        let probabilities = predictor.calculate(

            vector: vector

        )

        let tokenID = selector.select(

            probabilities: probabilities

        )

        let token = vocabulary.word(

            for: tokenID

        )

        return token
    }
}
