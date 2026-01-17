using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System;

public class EmotionChannel : SideChannel
{
    public float Valence { get; private set; }
    public float Arousal { get; private set; }
    public float Dominance { get; private set; }
    public string CurrentEmotionLabel { get; private set; }

    public EmotionChannel()
    {
        // Must match the UUID in Python
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f8");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        Valence = msg.ReadFloat32();
        Arousal = msg.ReadFloat32();
        Dominance = msg.ReadFloat32();
        CurrentEmotionLabel = msg.ReadString();

        Debug.Log($"[Emotion] {CurrentEmotionLabel} (V:{Valence:F2} A:{Arousal:F2} D:{Dominance:F2})");
        
        // Use this data to update visuals, e.g., facial expressions or lights
        UpdateVisuals();
    }

    private void UpdateVisuals()
    {
        // Placeholder for visual updates
        // Example: GetComponent<Renderer>().material.color = Color.Lerp(Color.blue, Color.red, (Valence + 1) / 2);
    }
}
