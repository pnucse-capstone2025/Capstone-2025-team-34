import string

system_message_for_train = (
    "You are a skilled English test-solving tutor.\n"
    "In the passage, any text referred to as 'underlined' is shown in bold. Example: **Was Underline**\n"
    "Read the problem and reason step by step.\n"
    "Use no more than 5 steps, shorter is better.\n"
    "max 50 characters per step, and write the final answer at the end."
)

user_prompt = """
[QUESTION]
다음 빈칸에 공통으로 들어갈 말로 가장 적절한 것을 고르시오.

[PASSAGE]
◦ She has a big smile on her _ .
◦ You should learn to _ your problem.

[OPTIONS]
A. face
B. heat
C. meet
D. walk

[ANSWER]
1. Sent1: smile on her ___ → noun
2. Sent2: ___ your problem → verb
3. B, C, D: don’t fit both
answer is: A

[QUESTION]
Human beings should care about the environment because _

[PASSAGE]
"Why should I care about the environment?" some people ask. There is a very simple reason: We live on the earth, and it is the only place we can live on right now, as we cannot live in space yet. If we treat the earth like a garbage dump, it will become dirty and unlivable. If we treat it well by being eco-conscious, the earth will stay clean and suitable for living, for ourselves and for our children.\nWhat is "being eco-conscious"?\nBeing eco-conscious means being aware of your actions, and what you do to the environment. For example, you might think, "Using hairspray is great for fixing my hair." However, if you are eco-conscious, you would ask, "Does this hairspray have CFCs in it? Could I be destroying the earth by using hairspray?"
How can I be eco-conscious?
You can find many simple ways to help the environment in your everyday life.
When you go shopping, bring a bag or a basket with you. Please try not to use plastic bags as much as you can.
You can go to work by bike or on foot if it is not far from your home.
You can put your rubbish into different bags, which makes it convenient for recycling.
You can eat less chewing gum.
You can use your own chopsticks instead of the disposable ones in your company or in restaurants.
You can put batteries in a special box instead of in garbage bins.
You can use the water which has been used for washing vegetables or fruits to clean the floor and the toilet or to water your garden.
You can turn off the decorative lights in your room when watching TV.
You can turn down your air conditioner by one degree, as this will cause a 10% reduction in energy use.\nYou can use recycled paper. Every ton of recycled office paper saves 380 gallons of oil. You can also use recycled glass instead of glass made from raw materials. This will reduce the related air pollution by 20%, and the related water pollution by 50%.\nPlease believe that. If you do a little, it will make a big difference.'

[OPTIONS]
A. if we make the earth unlivable, we'll have to live in space
B. it can help our children live better
C. if we don't, the earth will become a garbage dump
D. it will help to make the world a good living place

[ANSWER]
1. Question: why care for earth?
2. Main: keep earth livable place
3. A: space not possible
4. B, C: partial, not full reason
answer is: D

[QUESTION]
According to the passage, _ help children most.

[PASSAGE]
Most people want their children to be successful in school and a parent's role in that success must be very important. Parents should help children to build their confidence and achievements. Parents should also play the role of a friend as well as a teacher in children's education.
Learning about math or reading isn't the only problem we face in school. Maybe we are having problems with teachers, classmates or other people. At this time we really need a person who is ready to hear what we are thinking. And the person should be you---my dear parents. If we have nobody to talk with, we will have more stress in our school life. Please listen to our worries. It's good for our study and health.
On the other hand, parents can't leave all the problems to the teachers. Although it's their job, even the best and brightest teachers can not take care of every child. We still need personal attention, so the role the parents is to make sure we can get _ . Stay in touch with our homework and the tests. Connect with our teachers regularly to talk about how things are going in our classroom. If we need more help, be active in getting it and work with us.
Nothing helps a child succeed more than a parent. A little willingness from a parent can play a very important role in the children's studies. The more attention parents pay, the more achievements children will make.

[OPTIONS]
A. teachers
B. friends
C. parents
D. classmates

[ANSWER]
1. Keyword: help children most
2. Text: parents are key support
3. A: limited role
4. B, D: minor
answer is: C

[QUESTION]
{question}

[PASSAGE]
{article}

[OPTIONS]
{options_block}

[ANSWER]
"""

user_prompt_rational = """
[QUESTION]
다음 빈칸에 공통으로 들어갈 말로 가장 적절한 것을 고르시오.

[PASSAGE]
◦ She has a big smile on her _ .
◦ You should learn to _ your problem.

[OPTIONS]
A. face
B. heat
C. meet
D. walk

[ANSWER]
answer is A because
1. Sent1: smile on her ___ → noun
2. Sent2: ___ your problem → verb
3. B, C, D: don’t fit both
answer is: A

[QUESTION]
Human beings should care about the environment because _

[PASSAGE]
"Why should I care about the environment?" some people ask. There is a very simple reason: We live on the earth, and it is the only place we can live on right now, as we cannot live in space yet. If we treat the earth like a garbage dump, it will become dirty and unlivable. If we treat it well by being eco-conscious, the earth will stay clean and suitable for living, for ourselves and for our children.\nWhat is "being eco-conscious"?\nBeing eco-conscious means being aware of your actions, and what you do to the environment. For example, you might think, "Using hairspray is great for fixing my hair." However, if you are eco-conscious, you would ask, "Does this hairspray have CFCs in it? Could I be destroying the earth by using hairspray?"
How can I be eco-conscious?
You can find many simple ways to help the environment in your everyday life.
When you go shopping, bring a bag or a basket with you. Please try not to use plastic bags as much as you can.
You can go to work by bike or on foot if it is not far from your home.
You can put your rubbish into different bags, which makes it convenient for recycling.
You can eat less chewing gum.
You can use your own chopsticks instead of the disposable ones in your company or in restaurants.
You can put batteries in a special box instead of in garbage bins.
You can use the water which has been used for washing vegetables or fruits to clean the floor and the toilet or to water your garden.
You can turn off the decorative lights in your room when watching TV.
You can turn down your air conditioner by one degree, as this will cause a 10% reduction in energy use.\nYou can use recycled paper. Every ton of recycled office paper saves 380 gallons of oil. You can also use recycled glass instead of glass made from raw materials. This will reduce the related air pollution by 20%, and the related water pollution by 50%.\nPlease believe that. If you do a little, it will make a big difference.'

[OPTIONS]
A. if we make the earth unlivable, we'll have to live in space
B. it can help our children live better
C. if we don't, the earth will become a garbage dump
D. it will help to make the world a good living place

[ANSWER]
answer is D because
1. Question: why care for earth?
2. Main: keep earth livable place
3. A: space not possible
4. B, C: partial, not full reason
answer is: D

[QUESTION]
According to the passage, _ help children most.

[PASSAGE]
Most people want their children to be successful in school and a parent's role in that success must be very important. Parents should help children to build their confidence and achievements. Parents should also play the role of a friend as well as a teacher in children's education.
Learning about math or reading isn't the only problem we face in school. Maybe we are having problems with teachers, classmates or other people. At this time we really need a person who is ready to hear what we are thinking. And the person should be you---my dear parents. If we have nobody to talk with, we will have more stress in our school life. Please listen to our worries. It's good for our study and health.
On the other hand, parents can't leave all the problems to the teachers. Although it's their job, even the best and brightest teachers can not take care of every child. We still need personal attention, so the role the parents is to make sure we can get _ . Stay in touch with our homework and the tests. Connect with our teachers regularly to talk about how things are going in our classroom. If we need more help, be active in getting it and work with us.
Nothing helps a child succeed more than a parent. A little willingness from a parent can play a very important role in the children's studies. The more attention parents pay, the more achievements children will make.

[OPTIONS]
A. teachers
B. friends
C. parents
D. classmates

[ANSWER]
answer is C because
1. Keyword: help children most
2. Text: parents are key support
3. A: limited role
4. B, D: minor
answer is: C

[QUESTION]
{question}

[PASSAGE]
{article}

[OPTIONS]
{options_block}

[ANSWER]
answer is {answer} because
"""

user_prompt_train = """
[QUESTION]
{question}

[PASSAGE]
{article}

[OPTIONS]
{options_block}

[ANSWER]
"""

def format_options(opts):
    letters = string.ascii_uppercase 
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))

def create_conversation_for_generate(sample):
  opts = sample["options"]
  return {
    "messages": [
      {"role": "system", "content": system_message_for_train},
      {"role": "user", "content": user_prompt_train.format(article=sample["article"] , question=sample["question"], options_block=format_options(opts))}
    ]
  }
