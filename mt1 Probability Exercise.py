#!/usr/bin/env python
# coding: utf-8

# ## **Write your answers directly in the cells on this notebook.**
# 
# Use as name of the variables the ones provided as placeholders (often `sol=0`. Delete the `raise NotImplementedError()` line.
# 
# ### Show the entire process, from the initial fractions to the final result. 
# The probability results must be a decimal value between 0.0 and 1.0. Round your number to the third decimal (e.g 0.574).**

# Consider the following boxes (two boxes (red/blue) with two types of balls (green/orange)):
# ![image.png](image.png)

# #### Exercise 1: which is the probability of selecting the red box?

# In[4]:


sol = 0
sol = 1/2
sol


# In[2]:


assert sol == 0.5


# #### Exercise 2: if you want to run out of balls both boxes at the same time, how often should you select the blue box?

# In[3]:


sol = 0
# YOUR CODE HERE



sol = round(4/12, 3)
sol


# In[4]:


assert abs(sol-0.3333333333333333) < 0.01


# #### Exercise 3: which is the probability of picking up a green ball if you are using the red box?

# In[5]:


sol = 0
# YOUR CODE HERE
sol = round (2/8,3)
sol


# In[6]:


assert sol == 0.25


# #### Exercise 4: which is the probability of picking up an orange ball (any of the boxes)?

# In[6]:


sol = 0
# YOUR CODE HERE

sol= round(7/12, 3)
sol


# In[8]:


assert abs(sol-0.5833333333333334) < 0.01


# #### Exercise 5: given that you picked up a green ball, which is the probability of having selected the red box?

# In[24]:


sol = 0
# YOUR CODE HERE

sol = round(2/8)
sol


# In[10]:


assert abs(sol-0.125) < 0.3


# #### Optional Exercise 6: I have chosen a ball from a box and I picked up a green one. Which is the probability of having used the blue box?

# In[7]:


# No need to use the Bayes Theorem as it has not been explained in class
sol = 0
# YOUR CODE HERE
prob_green_bluebox = 3/5
prob_bluebox= 1/2
sol = prob_green_bluebox * prob_bluebox
sol


# In[12]:


assert abs(sol-0.375) < 0.9


# Now consider a standard deck of cards (the one used in Poker):
# ![deck.png](deck.png)
# 

# #### Exercise 7: which is the probability of drawing an Ace?

# In[10]:


# Create function that returns probability percent rounded to three decimal places

def event_probability(event_outcomes, sample_space):
    probability = 0
    probability = round(event_outcomes / sample_space, 3)

    return probability

sol = event_probability(4, 52)
    
sol


# In[14]:


# Using the previously defined function: Determine the probability of drawing an Ace
assert abs(sol-0.077 ) < 0.01


# #### Exercise 8: which is the probability of drawing a card that is a Heart?

# In[11]:


# Using the previously defined function: Determine the probability of drawing a heart

# YOUR CODE HERE

sol = event_probability(13, 52)
sol


# In[16]:


assert abs(sol-0.25 ) < 0.01


# #### Exercise 9: which is the probability of drawing a face card (such as Jacks, Queens, or Kings)?

# In[12]:


# Using the previously defined function: Determine the probability of drawing a face card

# YOUR CODE HERE
sol = event_probability(12, 52)
sol


# In[18]:


assert abs(sol-0.231 ) < 0.01


# #### Exercise 10: which is the probability of drawing a face card which is also a Heart?

# In[13]:


# Using the previously defined function: Determine the probability of drawing the jack, queen or king  of hearts
# YOUR CODE HERE

sol = event_probability(3, 32)
sol


# In[14]:


assert abs(sol-0.058 ) < 0.01


# #### Exercise 11: which is the probability of drawing an Ace on the second draw, if the first card drawn was a King?

# In[15]:


# Using the previously defined function: Determine the probability of drawing an Ace after drawing a King on the first draw


# YOUR CODE HERE
sol = event_probability(4, 51)
sol


# In[ ]:


assert abs(sol-0.078 ) < 0.01


# #### Exercise 12: which is the probability of drawing an Ace on the second draw, if the first card drawn was an Ace?

# In[16]:


# Using the previously defined function: Determine the probability of drawing an Ace after drawing an Ace on the first draw

# YOUR CODE HERE
sol = event_probability(3, 51)
sol


# In[ ]:


assert abs(sol-0.059) < 0.01


# Now consider the following situation:
# 
# You are playing Poker in the [Texas Holdem variant](https://en.wikipedia.org/wiki/Texas_hold_%27em). In case you are not familiar, it is a variant of poker in which each player has two cards and there is a set of community cards to play from. 
# ![texas.png](texas.png)

# #### Exercise 13: what is the probability that the next card drawn will be a Diamond card?

# In[17]:


# Sample Space
cards = 52
player_cards = 2
turn_community_cards = 4
# YOUR CODE HERE
remaining_cards = cards - player_cards - turn_community_cards

total_diamond_cards = 13
remaining_diamond_cards = 13 - 4

sol = event_probability (remaining_diamond_cards, remaining_cards)
sol


# In[ ]:


assert abs(sol-0.196) < 0.01


# Now consider the following situation:
# ![image.png](texas2.png)

# #### Exercise 14: which is the probability that with the next card drawn there will be five cards in sequential order? 
# 
# (Notice that any Eight ( 8, 9, 10, Jack, Queen) or any King (9, 10, Jack, Queen, King) will complete the straight)

# In[18]:


# Sample Space
cards = 52
player_cards = 2
turn_community_cards = 4

# YOUR CODE HERE

sol


# In[ ]:


assert abs(sol-0.174) < 0.01


# ## Now, we keep playing Texas Holdem 
# ![](https://media.giphy.com/media/EQBsWLfdxHW01Y1OHK/giphy.gif)
# ### But looking only at the cards we could have in our hand
# ### (2 possible drawing events, ignore community cards)
# #### Exercise 15: which is the probability of drawing a heart OR drawing a club?

# In[19]:


# Calculate the probability of drawing a heart or a club
# YOUR CODE HERE

sol = event_probability(26,cards) + event_probability(26,cards)*event_probability(26,cards-1)
sol = 1 - event_probability(26,cards)*event_probability(25,cards-1)

sol


# In[ ]:


assert abs(sol-0.755) < 0.01


# #### Exercise 16: which is the probability of drawing an ace, a king or a queen?

# In[21]:


# Calculate the probability of drawing an ace, king, or a queen
# YOUR CODE HERE

sol = event_probability(12,cards) + event_probability(40,cards)*event_probability(12,cards-1)
sol = 1 - event_probability(40,cards)*event_probability(39,cards-1)
sol


# In[ ]:


assert abs(sol-0.41171499999999994) < 0.01


# #### Exercise 17: which is the probability of drawing a heart or an ace?

# In[22]:


# Calculate the probability of drawing a heart or an ace
# YOUR CODE HERE
sol = event_probability(16,cards)+event_probability(36,cards)*event_probability(16,cards-1)
sol


# In[ ]:


assert abs(sol-0.525288) < 0.01


# #### Exercise 18: which is the probability of drawing a red card or drawing a face card?

# In[23]:


# Calculate the probability of drawing a red card or a face card

# YOUR CODE HERE
sol = event_probability(32,cards)+event_probability(20,cards)*event_probability(32,cards-1)
sol


# In[ ]:


assert abs(sol-0.856395) < 0.01


# #### Exercise 19: which is the probability of drawing an Ace from a deck of cards, replacing it, reshuffling the deck, and drawing another Ace?

# In[24]:


# Sample Space
cards = 52

# Outcomes
aces = 4

# YOUR CODE HERE
sol = event_probability(4,cards)*event_probability(4,cards)


sol


# In[ ]:


assert abs(sol-0.005929) < 0.001


# #### Exercise 20: which is the probability of being dealt two Aces (drawing one Ace after the other starting with a full deck)?

# In[25]:


# Sample Space first draw
cards = 52

# Outcomes first draw
aces = 4

# YOUR CODE HERE
sol = event_probability(4,cards)*event_probability(3,cards-1)
sol


# In[ ]:


assert abs(sol-0.004542999999999999) < 0.001


# ### Permutations and combinations

# #### Exercise 21: How many different 5-letter arrangements are there of the letters in the word morse?

# In[26]:


import math
sol = 0
# YOUR CODE HERE
sol = math.factorial(5)
sol


# In[ ]:


assert sol == 120


# #### Exercise 22: How many different seven-letter arrangements of the letters in the word HEXAGON can be made if each letter is used only once?

# In[27]:


# YOUR CODE HERE
sol = math.factorial(7)
sol


# In[ ]:


assert sol == 5040


# #### Exercise 23: Two cards are drawn at random from a standard deck of cards, without replacement. Find the probability of drawing an 8 and a queen in that order.

# In[28]:


# YOUR CODE HERE
sol = event_probability(4,52) * event_probability(4,51)
sol


# In[ ]:


assert abs(sol-0.006006) < 0.001

