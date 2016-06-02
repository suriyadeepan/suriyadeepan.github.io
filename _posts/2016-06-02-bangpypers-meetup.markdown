---
published: true
title: BangPypers Meetup
layout: post
bigimg: /img/meshnet/meshcover.jpg
subtitle: Talks on Mesh Networks
tags: [mesh, network, community, network]
---
> On Saturday 21st May, we travelled to Bangalore to deliver talks on Mesh Networks at [BangPypers Meetup](http://bangalore.python.org.in/about.html). Our friend Ramaseshan, deserves most of the credit for arranging the meet and accommodating us.  Overall, it was a successful event. Participants were excited to interact with us and learn about our activities. 

The session was split into 3 parts.

1. The Need for Mesh Network by *[Selva Kumar](https://github.com/vanangamudi)*
2. [Hardware for Mesh Networks](https://github.com/lrmodesgh/Presentation/blob/master/0%20Main%20Subjects/9%20Radio%20Mesh%20Networking%20%26%20Distributed%20Systems/Radio%20Mesh%20Networking%20and%20Distributed%20Systems.pdf) by *[Ganesh Gopal](https://github.com/lrmodesgh)*
3. [Mesh Network : Practical Guide](http://pymeshnet.gitlab.io/slides/bangpypermeet/software/) by *[Suriyadeepan Ramamoorthy](https://github.com/suriyadeepan)*

## The Need for Mesh Network

Selva explained how Internet works, on infrastructure level. He contrasted the Centralized and Decentralized systems and explained the need for decentralization on infrastructure level (Material P2P). He went on to explain the need for an alternative to IP, which was invented with hierarchical control in mind. He also explained how mesh networks (ad hoc networks) come in handy during natural disasters like earthquakes and floods. During the end of his talk, one person from the audience, asked if the network is owned and operated by the community without a central agent (think ISP), how can the government conduct surveillance? Selva replied that it is an ethical argument. Should the government spy on its people?

## Hardware for Mesh Networks

Ganesh took over and started with how our routers work and different parts of the router. He explained the working of an antenna, types of antennas and showed a few simulations of how different antennas work in different environments. He, then showed a list of RF experimentation/prototyping platforms. He mentioned that his interest in radio communication started when he experimented with Ham radio/Amateur radio.

## Mesh Network : Practical Guide

Finally, I had the stage to myself. I covered the software section, from Routing, OpenWrt to P2P applications. I explained how to setup B.A.T.M.A.N protocol in a linux system and then in a openwrt installed router. I described the problems we faced at [pymesh](http://pymeshnet.gitlab.io), like IP allocation and DNS, and how we solved them. I listed a few use cases of a community network and then showed a few P2P applications that can be deployed in the network. Later I explained zero conf, its advantages and how it works. At the end of the session, I mentioned zeronet briefly and urged them to look into it ASAP. 

**Conluding remarks**

> Mesh network is an amalgamation of diverse fields. Everyone has something in it. For developing applications in a mesh network, one must change his way of thinking about application development. One must rethink the entire process, keeping in mind the ideas like Decentralization, Distribution, Security and Zero Configuration. This is a great opportunity for developers to port the existing applications or rewrite the applications for this new breed of network that respects user's freedom.
