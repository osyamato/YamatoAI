import Foundation

struct YamatoModel {

    let vocabulary: Vocabulary

    let embedding: Embedding

    let positionalEncoding: PositionalEncoding

    let transformer: Transformer

    let decoder: TokenDecoder

    func predictNextToken(

        text: String

    ) -> String? {

        let tokenizer = Tokenizer()

        let tokens = tokenizer.tokenize(

            text

        )

        var tokenIDs: [Int] = []

        for token in tokens {

            guard let id = vocabulary.id(

                for: token

            ) else {

                return nil
            }

            tokenIDs.append(id)
        }
        
        var embeddings: [EmbeddingVector] = []

        for id in tokenIDs {

            guard let vector = embedding.vector(

                for: id

            ) else {

                return nil

            }

            embeddings.append(vector)

        }
        let encodedEmbeddings = positionalEncoding.calculate(

            embeddings: embeddings

        )
        
        
        
        let transformerResults = transformer.calculate(

            embeddings: encodedEmbeddings

        )
        guard let lastVector = transformerResults.last else {

            return nil

        }
        
        let nextToken = decoder.decode(

            vector: lastVector

        )

        return nextToken
    }
}


