import Foundation

struct EmbeddingVector {

    let values: [Float]

    static func random(dimension: Int) -> EmbeddingVector {

        let values = (0..<dimension).map { _ in

            Float.random(in: -1...1)

        }

        return EmbeddingVector(values: values)

    }

}
