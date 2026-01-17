using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System;
using System.Text;

public class ConsciousnessChannel : SideChannel
{
    public float CurrentPhi { get; private set; }
    public bool IsConscious { get; private set; }
    public string FocusContent { get; private set; }

    public ConsciousnessChannel()
    {
        // Must match the UUID in Python
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Read the message from Python
        CurrentPhi = msg.ReadFloat32();
        float consciousFlag = msg.ReadFloat32();
        IsConscious = consciousFlag > 0.5f;
        FocusContent = msg.ReadString();

        Debug.Log($"[Consciousness] Phi: {CurrentPhi}, Conscious: {IsConscious}, Focus: {FocusContent}");
    }

    public void SendState(string message)
    {
        // Example of sending data BACK to Python if needed
        using (var msg = new OutgoingMessage())
        {
            msg.WriteString(message);
            QueueMessageToSend(msg);
        }
    }
}
