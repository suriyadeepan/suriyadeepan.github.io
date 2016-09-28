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

Each message consists of a user name, time stamp, message text and a location, where user name and location are optional. In contrast to existing social networks, we encourage the users to stay anonymous by using pseudonyms as their handle. Users can also choose to post zombie messages i.e. a message without a handle. These zombie messages contain a time stamp, message text and optionally a location. Messages sent via SMS are by default zombie messages. The central server maintains a history of all the messages. This data is publicly available for anyone to access and analyze. 

The network model the enables offline users to share information, is the key idea in our proposal. It consists of a mesh network of **leaf** nodes. The leaf nodes act as the receptors that connect the user to the network. The leaf nodes are strategically placed around a gateway node, in order to maximize radio coverage. A gateway node connects the mesh network to the internet. The gateway node is also connected to a processor (**processing node**) that stores, processes and pushes bulk messages to the internet. A low power, low cost tiny computer like **beaglebone** can be used as the processing node. An SMS gateway is established that will accumulate messages received from mobile phones and pushes them to the central server.

An android application is provided for smartphone users, which serves as the client. It stores the messages from users and then pushes them to a leaf node when it comes in contact with one or directly to the server if connected to the internet. In addition to that, other smartphones that use the same application are used as data mules for storing and forwarding the users messages, without exposing them.


## Use cases

1. Report a flaw (Poorly laid roads, open drains, Improper Chemical waste disposal, etc,.)
2. Report an incident (Unauthorized cut down of trees, Noise pollution)

## Conclusion

The network that we are proposing should grow organically. Our objective is to kickstart this network of networks by bootstrapping the initial leaf nodes and to facilitate its growth.


## TODO

* [ ] Pictorial depiction of Architecture
* [ ] Image depicting overlay of mesh network over map
* [ ] Fabricated screenshot of Android application (client), Web Interface (dashboard) and SMS screen
* [ ] Project Planning : Phases of deployment (Refer to [ieg](https://meta.wikimedia.org/wiki/Grants:IEG/Wiki_Mesh_Network_Community))
* [ ] Maintenance : Through local communities
* [ ] Budget : AR-300, Beaglebone
* [ ] Geolocation Planning : Choose a specific region ideal for kickstarting (bootstrap) the project
