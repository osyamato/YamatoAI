import Foundation

struct TrainingExample: Codable {

    let input: String

    let output: String

    let category: Category

    let style: ResponseStyle

}

enum ResponseStyle: String, Codable {

    case direct       // 直接答える

    case gentle       // 優しく答える

    case reflective   // 考えさせる

    case angel        // 天使の企て

}

enum Category: String, Codable {

    case greeting

    case emotion

    case weather

    case animal

    case garden

    case philosophy

}
