#DICIONARIO DE PARAMENTROS
#%%
import datetime as datetime

def startDate(date):
    stdDate = date - datetime.timedelta(hours=3)
    return stdDate

    
now = datetime.datetime.now()
startDate(now)

#%%

endDate = 
    var startDt = !params.start ? new Date() : new Date(Date.parse(params.start + " 00:00:00.000-03:00"))
    var endDt = !params.end ? new Date() : new Date(Date.parse(params.end + " 23:59:59.999-03:00"))

    startDt = { "$toDate": startDt }
    endDt = { "$toDate": endDt }


    var match1 = {}
    var match2 = { "deals.status": { "$nin": ["pending", "error"] } }
    var match3 = {}
    var match4 = {}
    var project = {}

    switch (params.period_group) {
        case "Dia":
            params.period_group = "%Y-%m-%d";
            params.group = "$date"; break
        case "Mês":
            params.period_group = "%Y-%m";
            params.group = "$date"; break
        case "Mês Vencimento":
            params.period_group = "%Y-%m";
            params.group = "$dueAt"; break
        case "Ano":
            params.period_group = "%Y";
            params.group = "$date"; break
        case "Credor":
            params.period_group = "";
            params.group = "$creditor"; break
        case "Parcela":
            params.period_group = "";
            params.group = "$installment"; break
        case "Canal":
            params.period_group = "";
            params.group = "$channel"; break
        case "Credor-Canal":
            params.period_group = "";
            params.group = "$creditor_channel"; break
        case "Rank Divida":
            params.period_group = "";
            params.group = "$debt_tag"; break
        case "Tudo":
            params.period_group = "";
            params.group = null; break
        case "ID":
            params.period_group = "";
            params.group = "$_id"; break
    }

    switch (params.period_type) {
        case "Vencimento":
            match1["installments"] = { "$elemMatch": { "dueAt": { "$gte": startDt, "$lte": endDt } } }
            match4["installments.dueAt"] = { "$gte": startDt, "$lte": endDt }
            project = { "date": { "$dateToString": { "date": "$installments.dueAt", "format": params.period_group, "timezone": "-0300" } } }
            break;
        case "Pagamento":
            match1["installments"] = {
                "$elemMatch": {
                   /*"status": "paid",*/
                    "$or": [
                        { "payment.paidAt": { "$gte": startDt, "$lte": endDt } },
                        { "payment.paidAt": { "$eq": null }, "dueAt": { "$gte": startDt, "$lte": endDt } }
                    ]
                }
            }
            match4["$or"] = [
                { "installments.payment.paidAt": { "$gte": startDt, "$lte": endDt } },
                { "installments.payment.paidAt": { "$eq": null }, "installments.dueAt": { "$gte": startDt, "$lte": endDt } }
            ]
            project = { "date": { "$dateToString": { "date": { "$ifNull": ["$installments.payment.paidAt", "$installments.dueAt"] }, "format": params.period_group, "timezone": "-0300" } } }
            break;
        default:
            match1["deals"] = { "$elemMatch": { "createdAt": { "$gte": startDt, "$lte": endDt } } }
            match2["deals.createdAt"] = { "$gte": startDt, "$lte": endDt }
            project = { "date": { "$dateToString": { "date": "$installments.createdAt", "format": params.period_group, "timezone": "-0300" } } }
            break;
    }


    var creditorFirstFilter = { "$ne": "" }
    if (params.creditor && params.creditor.indexOf("Todos") == -1 && params.creditor != []) {
        if (match1["installments"]) {
            match1["installments"]["$elemMatch"]["creditor"] = { "$in": params.creditor }
        } else {
            match1["deals"]["$elemMatch"]["creditor"] = { "$in": params.creditor }
        }
        match4["installments.creditor"] = { "$in": params.creditor }
        match2["deals.creditor"] = { "$in": params.creditor }
        creditorFirstFilter = { "$in": params.creditor }
    }


    if (params.portfolio && params.portfolio.indexOf("Todos") == -1 && params.portfolio != []) {
        if(match1["deals"]){ 
           match1["deals"]["$elemMatch"]["offer.debts.portfolio"] = { "$in": params.portfolio }
        }else{
            match1["deals"]= {"$elemMatch" : { "offer.debts.portfolio" : { "$in": params.portfolio } } }
        }
        match2["deals.offer.debts.portfolio"] = { "$in": params.portfolio }
    }

    if (params.installments && params.installments.indexOf("Todas") == -1 && params.installments != []) {
        match1["installments"]["$elemMatch"]["installment"] = { "$in": params.installments.map(function (i) { return parseInt(i); }) }
        match4["installments.installment"] = { "$in": params.installments.map(function (i) { return parseInt(i); }) }
    }

    switch (params.totalInstallments) {
        case "Todos": params.totalInstallments = ["Todos"]; break;
        case "À vista": params.totalInstallments = ["1"]; break;
        case "Parcelado":
            params.totalInstallments = []
            for (var i = 2; i < 100; i++) {
                params.totalInstallments.push(i)
            }
            break;
    }

    if (params.totalInstallments && params.totalInstallments.indexOf("Todos") == -1 && params.totalInstallments != []) {
        match1["deals"] = { "$elemMatch": {} }
        match1["deals"]["$elemMatch"]["totalInstallments"] = { "$in": params.totalInstallments.map(function (i) { return parseInt(i); }) }
        match2["deals.totalInstallments"] = { "$in": params.totalInstallments.map(function (i) { return parseInt(i); }) }
    }


    switch (params.platform) {
        case "Nova":
            match2["deals.simulationID"] = { "$exists": true }; break;
        case "Antiga":
            match2["deals.simulationID"] = { "$exists": false }; break;
    }

    var channelFirstFilter = { "$ne": "" }
    if (params.channel && params.channel.indexOf("Todos") == -1 && params.channel != []) {
        match3["channel"] = { "$in": params.channel }
        channelFirstFilter = { "$in": params.channel }
    } else {
        match3["channel"] = { "$exists": true }
    }


    if (params.docType && params.docType != "Todos") {
        match1["documentType"] = { "$in": params.docType }
    }

    var checkAmount = { "$cond": [{ "$gt": ["$installments.payment.paidAmount", 0] }, "$installments.payment.paidAmount", "$installments.installmentAmount"] }

    var query = {
        aggregate: "col_person",
        cursor: { batchSize: 100000 },
        allowDiskUse: (params||{}).allowDiskUse||false,
        pipeline: [
            { "$match": { "deals": { "$elemMatch": { "createdAt": { "$lte": endDt }, "creditor": creditorFirstFilter/*, "channel": channelFirstFilter*/  } } } },
            { "$match": match1 },
            { "$unwind": "$deals" },
            { "$match": match2 },
            {
                "$addFields": {
                    "debt_tag": {
                        "$ifNull": [{
                            "$arrayElemAt": [
                                {
                                    "$reduce": {
                                        "input": {
                                            "$map": {
                                                "input": { "$ifNull": ["$deals.offer.debts", []] },
                                                "as": "d",
                                                "in": {
                                                    "$filter": {
                                                        "input": { "$ifNull": ["$$d.tags", []] },
                                                        "as": "d",
                                                        "cond": {
                                                            "$in": ["$$d", ["rank:a", "rank:b", "rank:c", "rank:d", "rank:e"]],
                                                        }
                                                    }
                                                },
                                            },
                                        },
                                        "initialValue": [],
                                        "in": { "$cond": [{ "$in": ["$$this", "$$value"] }, "$$value", { "$concatArrays": ["$$this", "$$value"] }] }
                                    }
                                }, 0]
                        }, "no_tag"]
                    },
                    "firstBrokenInstallment": {
                        "$reduce": {
                            "input": "$installments",
                            "initialValue": 99,
                            "in": {
                                "$min": ["$$value", {
                                    "$cond": [
                                        {
                                            "$and": [
                                                { "$eq": ["$$this.dealID", "$deals._id"] },
                                                { "$ne" :  ["$$this.status", "paid"] },
                                                //{ "$in": ["$$this.status", ["canceled", "broken", "expired"]] },
                                            ]
                                        },
                                        "$$this.installment", 99]
                                }]
                            }
                        }
                    },
                    "channel": {
                        "$cond": [
                            { "$ifNull": ["$deals.tracking.channel", false] },
                            "$deals.tracking.channel",
                            {
                                "$cond": [
                                    { "$ifNull": ["$deals.tracking.utms.source", false] },
                                    "$deals.tracking.utms.source",
                                    {
                                        "$cond": [
                                            { "$ifNull": ["$deals.offer.tokenData.channel", false] },
                                            "$deals.offer.tokenData.channel",
                                            { "$cond": [{ "$ifNull": ["$deals.simulationID", false] }, "web", "unknown"] }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            { "$match": match3 },
            { "$unwind": "$installments" },
            { "$match": match4 },
            {
                "$match": {
                    "$and": [
                        { "$expr": { "$eq": ["$deals._id", "$installments.dealID"] } },
                        { "$expr": { "$eq": ["$deals.creditor", "$installments.creditor"] } },
                        { "$expr": { "$gte": ["$deals.totalInstallments", "$installments.installment"] } },
                    ]
                }
            },
            {
                "$match": {
                    "$expr": { "$lte": ["$installments.installment", "$firstBrokenInstallment"] }
                }
            },
            { "$addFields": project },
            {
                "$project": {
                    "_id": "$installments._id",
                    "date": "$date",
                    "dueAt": { "$dateToString": { "date": "$installments.dueAt", "format": "%Y-%m", "timezone": "-0300" } },
                    "installment": "$installments.installment",
                    "creditor": "$installments.creditor",
                    "channel": "$channel",
                    "creditor_channel": { "$concat": ["$installments.creditor", ":", "$channel"] },
                    "debt_tag" : "$debt_tag",

                    "promise_overdue_qty": { "$cond": [{ "$eq": ["$installments.installment", 1] }, 1, 0] },
                    "promise_overdue_amount": { "$cond": [{ "$eq": ["$installments.installment", 1] }, "$installments.installmentAmount", 0] },
                    "promise_paid_qty": { "$cond": [{ "$and": [{ "$eq": ["$installments.installment", 1] }, { "$eq": ["$installments.status", "paid"] }] }, 1, 0] },
                    "promise_paid_amount": { "$cond": [{ "$and": [{ "$eq": ["$installments.installment", 1] }, { "$eq": ["$installments.status", "paid"] }] }, checkAmount, 0] },

                    "deal_overdue_qty": { "$cond": [{ "$gt": ["$installments.installment", 1] }, 1, 0] },
                    "deal_overdue_amount": { "$cond": [{ "$gt": ["$installments.installment", 1] }, "$installments.installmentAmount", 0] },
                    "deal_paid_qty": { "$cond": [{ "$and": [{ "$gt": ["$installments.installment", 1] }, { "$eq": ["$installments.status", "paid"] }] }, 1, 0] },
                    "deal_paid_amount": { "$cond": [{ "$and": [{ "$gt": ["$installments.installment", 1] }, { "$eq": ["$installments.status", "paid"] }] }, checkAmount, 0] },

                    "all_overdue_qty": { "$literal": 1 },
                    "all_overdue_amount": "$installments.installmentAmount",
                    "all_paid_qty": { "$cond": [{ "$eq": ["$installments.status", "paid"] }, 1, 0] },
                    "all_paid_amount": { "$cond": [{ "$eq": ["$installments.status", "paid"] }, checkAmount, 0] }
                }
            },
            {
                "$group": {
                    "_id": params.group,
                    "date": { "$first": "$date" },
                    "dueAt": { "$first": "$dueAt" },
                    "installment": { "$first": "$installment" },
                    "creditor": { "$first": "$creditor" },
                    "channel": { "$first": "$channel" },
                    "creditor_channel": { "$first": "$creditor_channel" },
                    "debt_tag": { "$first": "$debt_tag" },

                    "promise_overdue_qty": { "$sum": "$promise_overdue_qty" },
                    "promise_overdue_amount": { "$sum": "$promise_overdue_amount" },
                    "promise_paid_qty": { "$sum": "$promise_paid_qty" },
                    "promise_paid_amount": { "$sum": "$promise_paid_amount" },

                    "deal_overdue_qty": { "$sum": "$deal_overdue_qty" },
                    "deal_overdue_amount": { "$sum": "$deal_overdue_amount" },
                    "deal_paid_qty": { "$sum": "$deal_paid_qty" },
                    "deal_paid_amount": { "$sum": "$deal_paid_amount" },

                    "all_overdue_qty": { "$sum": "$all_overdue_qty" },
                    "all_overdue_amount": { "$sum": "$all_overdue_amount" },
                    "all_paid_qty": { "$sum": "$all_paid_qty" },
                    "all_paid_amount": { "$sum": "$all_paid_amount" }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "00 - Grupo": params.group,

                    "01 - Promessa Vencimento Qtd": "$promise_overdue_qty",
                    "02 - Promessa Vencimento Valor": "$promise_overdue_amount",
                    "03 - Promessa Pagamento Qtd": "$promise_paid_qty",
                    "04 - Promessa Pagamento Valor": "$promise_paid_amount",
                    "05 - Promessa Efetividade Qtd": { "$cond": [{ "$gt": ["$promise_overdue_qty", 0] }, { "$multiply": [100, { "$divide": ["$promise_paid_qty", "$promise_overdue_qty"] }] }, 0] },
                    "06 - Promessa Efetividade Valor": { "$cond": [{ "$gt": ["$promise_overdue_amount", 0] }, { "$multiply": [100, { "$divide": ["$promise_paid_amount", "$promise_overdue_amount"] }] }, 0] },

                    "07 - Acordo Vencimento Qtd": "$deal_overdue_qty",
                    "08 - Acordo Vencimento Valor": "$deal_overdue_amount",
                    "09 - Acordo Pagamento Qtd": "$deal_paid_qty",
                    "10 - Acordo Pagamento Valor": "$deal_paid_amount",
                    "11 - Acordo Efetividade Qtd": { "$cond": [{ "$gt": ["$deal_overdue_qty", 0] }, { "$multiply": [100, { "$divide": ["$deal_paid_qty", "$deal_overdue_qty"] }] }, 0] },
                    "12 - Acordo Efetividade Valor": { "$cond": [{ "$gt": ["$deal_overdue_amount", 0] }, { "$multiply": [100, { "$divide": ["$deal_paid_amount", "$deal_overdue_amount"] }] }, 0] },

                    "13 - Total Vencimento Qtd": "$all_overdue_qty",
                    "14 - Total Vencimento Valor": "$all_overdue_amount",
                    "15 - Total Pagamento Qtd": "$all_paid_qty",
                    "16 - Total Pagamento Valor": "$all_paid_amount",
                    "17 - Total Efetividade Qtd": { "$cond": [{ "$gt": ["$all_overdue_qty", 0] }, { "$multiply": [100, { "$divide": ["$all_paid_qty", "$all_overdue_qty"] }] }, 0] },
                    "18 - Total Efetividade Valor": { "$cond": [{ "$gt": ["$all_overdue_amount", 0] }, { "$multiply": [100, { "$divide": ["$all_paid_amount", "$all_overdue_amount"] }] }, 0] }
                }
            },
            {
                "$sort": { "00 - Grupo": 1 }
            }
        ]
    }

    if (params.splitted == "Sim") {
        query.pipeline = query.pipeline.concat([
            {
                "$project": {
                    "date": "$date",
                    "_id": 0,
                    "info": {
                        "$objectToArray": "$$ROOT"
                    }
                }
            },
            { "$unwind": "$info" },
            {
                "$project": {
                    "_id": 0,
                    "date": "$date",
                    "info": "$info.k",
                    "value": "$info.v"
                }
            },
            { "$match": { "info": { "$ne": "00 - Data" } } }

        ])
    }

    return query
}

function transform(result) {
    return result
}
