---
layout: page
title: Social Community Network
subtitle: A Social Network for the Common People
---
## Abstract

On a global scale, there exists a huge divide between people who have access to internet and those do not. In India, roughly 15% of the population has access to internet. As a result, 85% of the population is excluded from the digital world. They do not have a platform for free speech and debate.  We address the digital divide in India and propose a long-term solution that gradually bridges this divide. Our system, once implemented, will encourage people to take up an active role in the development of their community. We propose to do this by making it easier for people to discuss and share their opinion on issues and through incentives.  It will increase awareness among people, about regional conflicts by facilitating direct interaction with people from other communities, instead of involving third party commercial services, like newspapers, media, etc,. In a developing country, there are a large number of unresolved issues, like lack of infrastructure, poor sanitation, frequent power cuts, etc,. By reporting these issues and logging them in a transparent platform, it makes it easier for common people and the expert teams, to localize the problem and understand it and possibly lead to a solution. The system that we propose, is fundamentally a community network, jointly owned and maintained by the community, without a central authority. We believe, the proposed system if implemented carefully in an incremental fashion, can bring about significant social and economic changes in the society, by bridging people together.

## Motivation

Internet is a pipe dream for 85% of Indian people. This is mostly due to the fact that the infrastructure of internet worldwide, is owned and maintained by a small group of commercial organizations. Creating infrastructures for internet access in rural areas is not profitable, as most people will not be able to afford it. A solution to this, is introduction of community networks. A network jointly owned and maintained by people. People who are well off, can contribute, by donating routers and other network infrastructure. A peer with internet access can share it with others via this network. 

Using community networks, we can create awareness among people, about social issues and current affairs. This will increase people's participation in democracy. A platform is provided for people to talk about the challenges faced by common people **TODO: rephrase**. This platform opens doors to healthy debate online, which might lead to interesting results.

A developing country like India faces many challenges. Some of them are listed below.

1. Infrastructure : roads, highways, schools, primary healthcare
2. Access to Clean Water
3. Sanitation : Public health, diseases
4. Electricity : frequent power cuts
5. Unemployment
6. Industrial Waste Management
7. Garbage Disposal

When people start reporting (logging) the problems they face directly, instead of using a middleman like newspapers, media, etc, it becomes convenient for others to understand the problem, localize it and possibly solve it. This type of transparent collaborative, problem solving technique, will eventually create a rich collection of strategies that can be applied to different scenarios. The transparency of the whole process, enables anyone to contribute, thus leading to the most optimal solution.


The proposed system is built on top of the following principles.

1. Every individual voice deserves to be heard
2. People should play an active role in a democracy; more than just voting
3. Members of different communities should connect with each other directly, not through a medium

## Architecture

The proposed system consists of a central server that maintains the messages from users in a database. It provides a feed of recent messages. Multiples interfaces are provided for the clients to post messages. Users that have access to the internet, can directly post messages, via their dashboard. The users who do not have access to the internet, must find a nearby leaf node (explained in the next section) to post their messages using a client application in their smartphone. There is another interface to the server, where users with ordinary mobile phones can post a message, via SMS.

Each message consists of a user name, time stamp, message content and a location, where user name and location are optional. Multimedia content like images and videos are supported. In contrast to existing social networks, we encourage the users to stay anonymous by using pseudonyms as their handle. Users can also choose to post zombie messages i.e. a message without a handle. These zombie messages contain a time stamp, message content and optionally a location. Messages sent via SMS are by default zombie messages. The central server maintains a history of all the messages. This data is publicly available for anyone to access and analyze.

The network model the enables offline users to share information, is the key idea in our proposal. It consists of a mesh network of **leaf** nodes. The leaf nodes act as the receptors that connect the user to the network. The leaf nodes are strategically placed around a gateway node, in order to maximize radio coverage. A gateway node connects the mesh network to the internet. The gateway node is also connected to a processor (**processing node**) that stores, processes and pushes bulk messages to the internet. A low power, low cost tiny computer like **beaglebone** can be used as the processing node. An SMS gateway is established that will accumulate messages received from mobile phones and pushes them to the central server.

An android application is provided for smartphone users, which serves as the client. It stores the messages from users and then pushes them to a leaf node when it comes in contact with one or directly to the server if connected to the internet. In addition to that, other smartphones that use the same application are used as data mules for storing and forwarding the users messages, without exposing them.


## Use cases

From the initial deployment to full organic growth of the network, there are a great many number of use cases. Peers can report physical flaws like poorly laid roads, open drains, improper chemical waste disposal, etc, as soon as they encounter it. This report will contain a GPS location, a time stamp and a message with possibly few images. The message is stored in user's smartphone and it is pushed to the network once the phone comes in contact with a leaf node, or if it is connected to the internet. Similarly illegal activities can be reported, like unauthorized cut down of trees, hunting of birds, etc,. Once these incidents pile up, a report can be made summarizing the events and identifying the location of incidents. Based on which, the perpetrators can be identified and held accountable and preventive measures can be taken. 

Apart from these, there are more complex issues, like construction of nuclear power plants, which call for rigorous discussion considering not just the pros and cons, but also ecological and ethical concerns. Our system will provide a platform for people to engage in healthy criticism of government policies and debate complex social issues. This will increase awareness among people as they are directly involved. 

Using Machine Learning techniques like Sentiment Analysis and Topic Modeling, we can analyze the textual information from the database, to auto-generate reports on critical issues. Along with Geographical Information System, we can use these reports to create and maintain records that are rich in information, to track the overall progress of a region.

## Evolution of Network

It is impossible for any non-profit organization to compete with the ISPs. Our network does not try to replace the existing infrastructure. The goal of our project is to kickstart a network of community networks by bootstrapping the initial leaf nodes and facilitate the organic growth of the network. The growth of the network is depicted in figure below **TODO : need figure below**. In the initial phase, we deploy one or two leaf nodes in a region, connected to a gateway node. We then, conduct a local community meeting, in which we introduce our system and its use cases. We encourage people to get involved in the network, by using it and hosting their own leaf nodes, by providing special incentives, like domain names, hosting services, etc, to node owners. As more users become node owners, the network grows gradually and connects with neighboring networks. We believe, this type of natural growth of the network, will eventually enable it to become standalone, without the need of multiple gateways. Such a massive mesh network raises several challenges like scalability and performance, which need to be tackled head on. But it also provides a valuable infrastructure, which offers infinite possibilities for its users.


## Summary

This project is an attempt to create a network that realizes the vision [[article](https://www.internetsociety.org/sites/default/files/the-internet-is-for-everyone.pdf)] of Vint Cerf, one of the fathers of Internet, an internet for everyone - affordable, unrestricted and unregulated. We aim to remove the middleman - ISPs, from the network and connect people with each other. We propose to achieve this by focusing on the socioeconomic issues influencing the society and by empowering local communities to report, understand and solve their own issues.
