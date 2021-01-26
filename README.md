#项目背景 
##1.比赛地址
https://zindi.africa/competitions/zimnat-insurance-recommendation-challenge
##2.介
       为了使保险市场运作良好，保险公司需要能够在一个广泛的客户基础上共享和分散风险。在参保人口多样化且人数众多的地区，这种方法效果最好。
    在非洲，由于缺乏提供保险的私营部门公司，无法实现人口风险的多样化和集中，因此，正式的风险保险受到了阻碍。了解不同人群的保险需求，并将
    其与保险公司提供的合适产品相匹配，可以使保险更有效，使保险公司更成功。其核心是，了解保险产品的消费者有助于保险公司完善、多样化和推广
    其产品。增加的数据收集和改进的数据科学工具提供了极大地提高这种理解的机会。
       在本次竞争中，您将利用数据和ML方法，通过将津巴布韦保险市场的消费者需求与产品提供相匹配，为保险供应商Zimnat改善市场结果。Zimnat
    希望使用ML模型来使用客户数据来预测向客户推荐哪种保险产品。该公司提供了近4万名从Zimnat购买过两种或两种以上保险产品的客户的数据。您的
    挑战:对于测试集中的大约10,000名客户，您将获得他们拥有的除一种以外的所有产品，并要求您预测哪些产品最有可能成为丢失的产品。然后可以将
    相同的模型应用于任何客户，以确定根据其当前配置文件可能对其有用的保险产品。
       为了参加这个比赛，我们要求你完成一份关于你作为数据科学家经历的调查。我们将在比赛过程中通过电子邮件发送调查结果。所有的调查数据将被
    匿名，并且只会被研究单位用于质量AI扩散的研究目的。
       自1946年以来，津纳特一直是津巴布韦人寿保险和短期保险行业的领军人物。70多年来，Zimnat一直在保护津巴布韦人的资产，管理他们的财富，
    并确保他们的资产和资金能够代代相传，如果他们愿意的话。
##要求
       对于训练和测试，每一行都对应一个客户，分配一个唯一的客户ID (' ID ')。有一些关于客户的信息(他们加入的时间，出生年份等)。还提供了
    客户的职业(' occupational _code ')和职业类别(' occupational _category_code ')，以及他们访问的办公室的分支代码。最后的21
    栏对应21种商品。在培训中，客户拥有的每个产品的相关栏都有一个1。测试与此类似，除了每个客户都有一个产品被删除(用0替换1)，您的目标是构
    建一个模型来预测丢失的产品。样例显示所需的提交格式。对于每个客户ID和每个产品，您必须预测该产品是客户正在使用的产品的可能性。请注意，示
    例提交包含测试集中包含的产品的1s，以便您可以关注未知的产品。
       将您的预测作为概率保留在0到1之间，不要四舍五入到0或1。
##变量描述
ID - 唯一用户id  
join_date - 加入保险的日期  
sex - 性别  
marital_status - 婚否  
birth_year - 出生年份  
branch_code - 客户注册的分支机构  
occupation_code - 描述客户端操作的代码  
occupation_category_code - 客户的工作所属类别  
P5DA - product code  
RIBP - product code  
8NN1 - product code  
7POT - product code  
66FJ - product code  
GYSR - product code  
SOP4 - product code  
RVSZ - product code  
PYUQ - product code  
LJR9 - product code  
N2MW - product code  
AHXO - product code  
BSTQ - product code  
FM3X - product code  
K6QO - product code  
QBOL - product code  
JWFN - product code  
JZ9D - product code  
J9JW - product code  
GHYX - product code  
ECY3 - product code  

# 算法介绍
## catboost
和lightgbm、xgboost并成为gbdt三大主流神器库，它是一种基于对称决策树（oblivious trees）
算法的参数少、支持类别型变量和高准确性的GBDT框架，主要解决的痛点是高效合理地处理类别型特征，
另外提出了新的方法来处理梯度偏差（Gradient bias）以及预测偏移（Prediction shift）问题，
提高算法的准确性和泛化能力:
## 一些策略
