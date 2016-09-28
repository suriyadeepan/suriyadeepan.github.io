---
layout: page
title: Social Community Network
subtitle: A Social Network for the Common People
---
## Abstract

The social networking sites like Twitter, Facebook and Instagram are flourishing throughout the world. Along with the growth of social networks, the number of researchers involved in Social Network Analysis has increased. Opinion Mining in Twitter data is a hot topic in NLP and Machine Learning, as 98% of twitter data is in public domain and easily available. Researchers have analyzed people's opinion on complex issues like Syria Refugee Crisis, Brexit, etc,. Although the existing social networks have proved as valuable tools, for people to express their opinions, it is inaccurate to claim that the social networks reflect the opinion of the general population. On a global scale, there exists a huge divide between people who have access to internet and those do not. In India, roughly 15% of the population has access to internet. As a result, 85% of the population is excluded from the digital world. They do not have a platform for free speech and debate.  We address the digital divide in India and propose a long-term solution that gradually bridges this divide. Our system, once implemented, will encourage people to take up an active role in the development of their community. We propose to do this by making it easier for people to discuss and share their opinion on issues and through incentives.  It will increase awareness among people, about regional conflicts by facilitating direct interaction with people from other communities, instead of involving third party commercial service, like newspapers, media, etc,. In a developing country, there are a large number of unresolved issues, like lack of infrastructure, poor sanitation, frequent power cuts, etc,. By reporting these issues and logging them in a transparent platform, it makes it easier for common people and the expert teams, to localize the problem and understand it and possibly lead to a solution. The system that we propose, is fundamentally a community network, jointly owned and maintained by the community, without a central authority. We believe, the proposed system if implemented carefully in an incremental fashion, can bring about significant social and economic changes in the society, by bridging people together.


## Introduction



I recently studied a few papers on Sentiment Analysis on Twitter data. Most of the researcher in the field of NLP, feel that Twitter is a gold mine for opinion mining. Because it reflects the mind set of the general population. Twitter serves as a safe bubble for people to exercise their right to free speech. By looking at the twitter feed for hash tags, say for example **#iphone7**, we can get an idea whether Apple is offering what they advertised. People are honest and carefree on twitter. And that is the essence of free speech.

Twitter data has also proved as a valuable tool for providing critical information during disasters like Earthquakes, floods, etc,. The Disaster Management Services have made efficient use of twitter feeds, to make plans to mitigate the effect of disasters.

The claim that twitter reflects the opinion of the people, is not entirely accurate. There is a huge divide between people who have access to internet and those who do not. It is safe to assume that twitter reflects the opinion of the urban folks. The voices of rural people go unheard. The objective of this project is to empower the rural people to participate in free speech.

## Motivation

This project is based on the following principles.

1. Every individual voice deserves to be heard
2. People should play an active role in a democracy; more than just voting
3. Members of different communities should connect with each other directly, not through a medium

## Anonymity

There are incidents in India, where people are arrested for criticizing politicians on Facebook. This calls for anonymity. Every whistle-blower will be ensured anonymity.

> Give a man a mask and he will tell you truth. 

## Problems facing Developing Countries

1. Infrastructure : roads, highways, schools, primary healthcare
2. Access to Clean Water
3. Sanitation : Public health, diseases
4. Electricity : frequent power cuts
5. Unemployment
6. Industrial Waste Management
7. Garbage Disposal
8. Frequent accidents : Risk zones


## Solutions

How does the government solve a problem? A think tank is formed : A team of experts who study the problem and provide an optimal solution. The knowledge of the common people is more valuable than any expert. The local people not only understand the problem, they also understand the circumstances and consequences. We propose a common platform, where people are encouraged to discuss existing social problem and propose their solutions collaboratively. This solution will provide useful insights to the team of experts and help them implement a solution efficiently, with the consent of the local people.

## Use cases

1. Report a flaw (Poorly laid roads, open drains, Improper Chemical waste disposal, etc,.)
2. Report an incident (Unauthorized cut down of trees, Noise pollution)

## Architecture

### Medium

1. Internet
2. SMS
3. Community Networks

### Network Model

1. Leaf Nodes <- Mobile phones/Commons Infrastructure
2. Processing Nodes (Beaglebone)
3. Gateway Nodes -> Internet

### Data model

1. Timestamp
2. Location : GPS coordinates [ municipality, counselor ]
3. Flaw/Incident

### Common Platform

Stack overflow like Social Networking application.


> People who understand the social problems and are willing to talk about it, do not have a proper platform available for sharing. The primary goal of this project, is to empower the community minded <*rephrase*> individuals and make sure that their voice is heard.

The network that we are proposing should grow oganically. Our objective is to kickstart this network of networks by bootstraping the initial leaf nodes and to facilitate this growth.


## TODO

* [ ] Localization
* [ ] Administrative Model of Puducherry
