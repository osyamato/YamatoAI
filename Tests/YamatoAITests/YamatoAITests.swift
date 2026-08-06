import XCTest
@testable import YamatoAI

final class YamatoAITests: XCTestCase {


    func testDotProduct() {

        let rabbit = EmbeddingVector(

            values: [1,2,3]

        )

        let dog = EmbeddingVector(

            values: [4,5,6]

        )

        let dotProduct = DotProduct()

        let result = dotProduct.calculate(

            between: rabbit,

            and: dog

        )

        XCTAssertEqual(result, 32)

    }
    
    func testSoftmax() {

        let softmax = Softmax()

        let scores: [Float] = [
            3,
            4,
            2,
            1
        ]

        let result = softmax.calculate(
            scores: scores
        )

        XCTAssertEqual(result[0], 0.2369, accuracy: 0.0001)

        XCTAssertEqual(result[1], 0.6439, accuracy: 0.0001)

        XCTAssertEqual(result[2], 0.0871, accuracy: 0.0001)

        XCTAssertEqual(result[3], 0.0321, accuracy: 0.0001)

    }
    
    func testMatrix() {

        let matrix = Matrix(

            values: [

                [1, 2],

                [3, 4]

            ]

        )

        XCTAssertEqual(matrix.values[0][0], 1)
        XCTAssertEqual(matrix.values[0][1], 2)

        XCTAssertEqual(matrix.values[1][0], 3)
        XCTAssertEqual(matrix.values[1][1], 4)

    }
    
    func testMatrixMultiply() {

        let matrix = Matrix(

            values: [

                [1, 2],

                [3, 4]

            ]

        )

        let vector = EmbeddingVector(

            values: [

                2,

                3

            ]

        )

        let result = matrix.multiply(

            vector: vector

        )

        XCTAssertEqual(

            result.values,

            [11, 16]

        )

    }
    
    func testAttention() {

        let embedding = EmbeddingVector(
            values: [1, 2]
        )

        let wq = Matrix(
            values: [
                [1,0],
                [0,1]
            ]
        )

        let wk = Matrix(
            values: [
                [1,0],
                [0,1]
            ]
        )

        let wv = Matrix(
            values: [
                [1,0],
                [0,1]
            ]
        )

        let attention = Attention()

        let result = attention.calculate(
            embedding: embedding,
            wq: wq,
            wk: wk,
            wv: wv
        )

        XCTAssertEqual(
            result.values,
            [1,2]
        )
    }
    
    

}
